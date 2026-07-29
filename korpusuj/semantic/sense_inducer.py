import math
import random
from collections import defaultdict
import numpy as np
import networkx as nx


class SenseInducer:
    """
    Kompatybilna wstecz wersja inducera:
    - nadal zwraca sense_id, members, vector
    """

    STOP_WORDS = {
        "być", "się", "i", "w", "z", "na", "o", "że", "który", "ten",
        "jak", "do", "nie", "co", "dla", "od", "za", "po", "to", "czy",
        "móc", "mieć", "zostać", "swój", "taki", "bardzo", "jako", "a", "ale"
    }




    DEFAULT_MAX_NEIGHBORS = 80
    DEFAULT_SIM_THRESHOLD = 0.62
    SECOND_HOP_PER_SEED = 8
    SECOND_HOP_MIN_SIM = 0.52
    MAX_DEGREE_PER_NODE = 8
    MIN_CLUSTER_SIZE = 3


    @staticmethod
    def cosine_sim(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    @staticmethod
    def chinese_whispers(G, iters=20, seed=42):
        rng = random.Random(seed) if seed is not None else random
        labels = {n: n for n in G.nodes()}
        nodes = list(G.nodes())

        for _ in range(iters):
            rng.shuffle(nodes)
            for n in nodes:
                scores = defaultdict(float)
                for nb, data in G[n].items():
                    scores[labels[nb]] += data.get("weight", 1.0)
                if scores:
                    labels[n] = max(scores.items(), key=lambda x: x[1])[0]

        clusters = defaultdict(list)
        for n, lab in labels.items():
            clusters[lab].append(n)
        return list(clusters.values())

    @classmethod
    def _apply_mmr(cls, v_center, candidate_words, vectors_dict, max_neighbors=50, lambda_param=0.62):
        selected = []
        unselected = candidate_words.copy()

        while len(selected) < max_neighbors and unselected:
            best_score = -float("inf")
            best_word = None

            for word in unselected:
                v_word = vectors_dict[word]
                sim_to_center = cls.cosine_sim(v_center, v_word)

                if not selected:
                    score = sim_to_center
                else:
                    sim_to_selected = max(
                        cls.cosine_sim(v_word, vectors_dict[sw]) for sw in selected
                    )
                    score = (lambda_param * sim_to_center) - ((1 - lambda_param) * sim_to_selected)

                if score > best_score:
                    best_score = score
                    best_word = word

            selected.append(best_word)
            unselected.remove(best_word)

        return selected

    @classmethod
    def _collect_multihop_pool(
        cls,
        lemma,
        vectors_dict,
        semantic_index,
        max_neighbors,
        debug=False
    ):
        if lemma not in vectors_dict or lemma not in semantic_index:
            return [], {}

        v_center = vectors_dict[lemma]
        raw_neighbors = semantic_index.get(lemma, [])[: max_neighbors * 4]

        one_hop_candidates = []
        for w, sim, freq in raw_neighbors:
            if w not in vectors_dict:
                continue
            if w.lower() in cls.STOP_WORDS:
                continue
            if w.lower() == lemma.lower():
                continue
            if float(sim) < 0.34:
                continue
            one_hop_candidates.append(w)

        if not one_hop_candidates:
            return [], {}

        one_hop = cls._apply_mmr(
            v_center=v_center,
            candidate_words=one_hop_candidates,
            vectors_dict=vectors_dict,
            max_neighbors=min(max_neighbors, len(one_hop_candidates)),
            lambda_param=0.62
        )

        support = defaultdict(float)
        support_meta = defaultdict(lambda: {
            "from_center": 0.0,
            "from_neighbors": 0.0,
            "neighbor_hits": 0,
            "raw_freq": 0
        })

        for w, sim, freq in semantic_index.get(lemma, [])[: max_neighbors * 4]:
            if w in one_hop:
                support[w] += float(sim)
                support_meta[w]["from_center"] = float(sim)
                support_meta[w]["raw_freq"] = int(freq or 0)

        for seed in one_hop:
            for u, sim2, freq2 in semantic_index.get(seed, [])[: cls.SECOND_HOP_PER_SEED]:
                if u == lemma:
                    continue
                if u not in vectors_dict:
                    continue
                if u.lower() in cls.STOP_WORDS:
                    continue
                if float(sim2) < cls.SECOND_HOP_MIN_SIM:
                    continue

                local_bonus = float(sim2) * 0.65
                support[u] += local_bonus
                support_meta[u]["from_neighbors"] += local_bonus
                support_meta[u]["neighbor_hits"] += 1
                support_meta[u]["raw_freq"] = max(
                    support_meta[u]["raw_freq"], int(freq2 or 0)
                )

        pool = []
        for w, score in support.items():
            if w.lower() == lemma.lower():
                continue
            if score <= 0:
                continue

            from_center = support_meta[w]["from_center"]
            neighbor_hits = support_meta[w]["neighbor_hits"]

            if from_center == 0.0 and neighbor_hits < 2:
                continue

            pool.append(w)

        pool = sorted(
            pool,
            key=lambda w: (
                support_meta[w]["from_center"] + support_meta[w]["from_neighbors"],
                support_meta[w]["neighbor_hits"],
                -support_meta[w]["raw_freq"],
                w
            ),
            reverse=True
        )

        hard_cap = min(len(pool), max_neighbors + (max_neighbors // 2))
        pool = pool[:hard_cap]

        if debug:
            print(f"[PUD] {lemma} :: one_hop={len(one_hop)} pool={len(pool)}")

        return pool, support_meta

    @classmethod
    def _node_background_hint(cls, word):
        w = (word or "").lower()
        if w in cls.LIGHT_BACKGROUND_MARKERS:
            return 1.00
        if w in cls.DISCOURSE_MARKERS:
            return 0.85
        return 0.0

    @classmethod
    def _build_frame_graph(
        cls,
        lemma,
        pool,
        vectors_dict,
        semantic_index,
        support_meta,
        sim_threshold,
        debug=False
    ):
        G = nx.Graph()
        if lemma not in vectors_dict:
            return G

        v_center = vectors_dict[lemma]

        for w in pool:
            if w not in vectors_dict:
                continue

            center_sim = support_meta[w]["from_center"]
            if center_sim == 0.0:
                center_sim = cls.cosine_sim(v_center, vectors_dict[w])

            genericity_penalty = math.log1p(support_meta[w]["raw_freq"] or 0) / 14.0

            background_hint = 0.0
            # Podnosimy center_sim na < 0.55. Dzięki temu słowa jak "powiedzieć",
            # "dodać", "wtorek" znów zostaną trafnie rozpoznane jako tło informacyjne.
            if genericity_penalty > 0.45 and center_sim < 0.55:
                background_hint = 0.85
            if genericity_penalty > 0.55 and center_sim < 0.45:
                background_hint = 1.00

            G.add_node(
                w,
                center_sim=float(center_sim),
                raw_freq=int(support_meta[w]["raw_freq"] or 0),
                support_from_neighbors=float(support_meta[w]["from_neighbors"]),
                neighbor_hits=int(support_meta[w]["neighbor_hits"]),
                genericity_penalty=float(genericity_penalty),
                background_hint=float(background_hint)
            )

        sorted_pool = sorted(G.nodes())
        pool_set = set(sorted_pool)

        for w in sorted_pool:
            for u, sim, freq in semantic_index.get(w, [])[:25]:
                if u not in pool_set:
                    continue
                if u == w:
                    continue

                support_bonus = 0.03 * min(
                    G.nodes[w]["neighbor_hits"],
                    G.nodes[u]["neighbor_hits"]
                )

                penalty = 0.45 * (
                    G.nodes[w]["genericity_penalty"] + G.nodes[u]["genericity_penalty"]
                )

                bg_penalty = 0.14 * (
                    G.nodes[w]["background_hint"] + G.nodes[u]["background_hint"]
                )

                adjusted = float(sim) + support_bonus - penalty - bg_penalty

                if adjusted >= sim_threshold:
                    G.add_edge(w, u, weight=float(adjusted))

        if G.number_of_edges() < max(1, len(sorted_pool) // 10):
            for i in range(len(sorted_pool)):
                for j in range(i + 1, len(sorted_pool)):
                    w, u = sorted_pool[i], sorted_pool[j]
                    sim = cls.cosine_sim(vectors_dict[w], vectors_dict[u])
                    if sim >= max(0.46, sim_threshold - 0.08):
                        if not G.has_edge(w, u):
                            # fallback też lekko karzemy za metadyskurs
                            adjusted = sim - 0.08 * (
                                G.nodes[w]["background_hint"] + G.nodes[u]["background_hint"]
                            )
                            G.add_edge(w, u, weight=float(adjusted))

        max_degree_per_node = cls.MAX_DEGREE_PER_NODE
        edges_to_remove = set()

        for node in list(G.nodes()):
            nbrs = sorted(
                G[node].items(),
                key=lambda x: x[1].get("weight", 0.0),
                reverse=True
            )
            for nb, _ in nbrs[max_degree_per_node:]:
                edges_to_remove.add(tuple(sorted((node, nb))))

        for u, v in edges_to_remove:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

        isolates = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolates)

        if debug:
            print(
                f"[PUD] {lemma} :: graph_nodes={G.number_of_nodes()} "
                f"graph_edges={G.number_of_edges()}"
            )

        return G

    @classmethod
    def _score_frame_members(cls, cluster, G):
        sub = G.subgraph(cluster).copy()
        if not sub.nodes:
            return [], [], []

        deg = dict(sub.degree(weight="weight"))
        sorted_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)

        anchors_scored = []
        bridges_scored = []
        background_candidates = []

        for node, local_deg in sorted_deg:
            nd = sub.nodes[node]

            anchor_score = (
                1.20 * nd.get("center_sim", 0.0)
                + 0.85 * local_deg
                + 0.25 * nd.get("neighbor_hits", 0)
                - 0.95 * nd.get("genericity_penalty", 0.0)
                - 1.00 * nd.get("background_hint", 0.0)
            )

            bridge_score = (
                0.55 * local_deg
                + 0.75 * nd.get("support_from_neighbors", 0.0)
                - 0.35 * nd.get("center_sim", 0.0)
            )

            background_score = (
                1.00 * nd.get("background_hint", 0.0)
                + 0.80 * nd.get("genericity_penalty", 0.0)
                - 0.35 * nd.get("center_sim", 0.0)
            )

            anchors_scored.append((node, anchor_score))
            bridges_scored.append((node, bridge_score))
            background_candidates.append(
                (node, background_score, nd.get("background_hint", 0.0), nd.get("center_sim", 0.0))
            )

        anchors = [w for w, _ in sorted(anchors_scored, key=lambda x: x[1], reverse=True)]
        bridges = [w for w, _ in sorted(bridges_scored, key=lambda x: x[1], reverse=True)]

        background = [
            w for w, score, hint, center_sim in sorted(
                background_candidates,
                key=lambda x: x[1],
                reverse=True
            )
            if (hint >= 0.85) or (score > 0.90 and center_sim < 0.45)
        ]

        return anchors, bridges, background

    @classmethod
    def _cluster_cohesion(cls, cluster, G):
        sub = G.subgraph(cluster)
        if sub.number_of_nodes() <= 1:
            return 0.0
        weights = [d.get("weight", 0.0) for _, _, d in sub.edges(data=True)]
        if not weights:
            return 0.0
        return float(np.mean(weights))

    @classmethod
    def _frame_type(cls, members, anchors, background_markers, G):
        """
        Ostrożniejszy podział:
        - semantic   = rama semantyczna
        - contextual = Rama kontekstowa

        Zmiana względem starej wersji:
        nie oznaczamy ramy jako 'contextual' tylko dlatego, że ma sporo tła,
        jeśli jej topowe anchory są wyraźnie semantyczne.
        """
        cluster_size = max(1, len(members))
        bg_ratio = len(background_markers) / cluster_size

        top_anchors = anchors[:4]

        anchor_bg_count = sum(
            1 for a in top_anchors
            if a in G.nodes and G.nodes[a].get("background_hint", 0.0) >= 0.85
        )

        anchor_sem_count = sum(
            1 for a in top_anchors
            if a in G.nodes and G.nodes[a].get("background_hint", 0.0) < 0.85
        )

        mean_center = float(np.mean([
            G.nodes[m].get("center_sim", 0.0)
            for m in members
            if m in G.nodes
        ])) if members else 0.0

        mean_genericity = float(np.mean([
            G.nodes[m].get("genericity_penalty", 0.0)
            for m in members
            if m in G.nodes
        ])) if members else 0.0

        # Najpierw silny sygnał semantyczny:
        # jeśli rama ma co najmniej 2 sensowne anchory i nie leży daleko od centrum,
        # nie oznaczamy jej jako dyskursywnej.
        if anchor_sem_count >= 2 and mean_center >= 0.55:
            return "semantic"

        # Silny sygnał dyskursowy:
        if bg_ratio >= 0.75:
            return "contextual"

        if anchor_bg_count >= 3:
            return "contextual"

        if bg_ratio >= 0.60 and anchor_bg_count >= 2:
            return "contextual"

        if mean_genericity >= 0.80 and mean_center < 0.55:
            return "contextual"

        return "semantic"

    @classmethod
    def _label_terms(cls, members, anchors, G, max_terms=3, allow_background=False):
        """
        Wybiera terminy do etykiety z preferencją dla anchorów,
        ale odrzuca słowa mocno tła/dyskursowe.
        """
        scored = []
        seen = set()

        ordered = list(anchors) + [m for m in members if m not in anchors]

        for word in ordered:
            if word in seen:
                continue
            if word not in G.nodes:
                continue

            seen.add(word)
            nd = G.nodes[word]

            bg = float(nd.get("background_hint", 0.0))
            gen = float(nd.get("genericity_penalty", 0.0))

            # domyślnie nie używamy słów tła jako labeli
            if not allow_background and bg >= 0.85:
                continue

            # odetnij najbardziej ogólne krótkie huby
            if gen >= 0.95 and len(str(word)) <= 3:
                continue

            score = (
                    2.00 * (1.0 if word in anchors[:3] else 0.0)
                    + 1.10 * nd.get("center_sim", 0.0)
                    + 0.35 * G.degree(word, weight="weight")
                    - 1.25 * bg
                    - 0.50 * gen
            )

            scored.append((word, float(score)))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:max_terms]]

    @classmethod
    def _frame_label(cls, members, anchors, background_markers, G, frame_type="semantic"):
        """
        Zwraca czysty label ramy (bez prefiksów typu 'dyskurs:').
        Typ ramy powinien być wyświetlany dopiero w UI / semantic report
        na podstawie pola `frame_type`.
        """
        if not anchors and not members:
            return "nieokreślona"

        terms = cls._label_terms(
            members=members,
            anchors=anchors,
            G=G,
            max_terms=3,
            allow_background=False
        )

        if len(terms) >= 2:
            return ", ".join(terms)

        fallback = anchors or members or background_markers
        return ", ".join(fallback[:3])



    @classmethod
    def induce(
        cls,
        lemma,
        vectors_dict,
        semantic_index,
        max_neighbors=None,
        sim_threshold=None,
        min_cluster_size=None,
        debug=False
    ):
        if lemma not in vectors_dict:
            return []

        max_neighbors = max_neighbors or cls.DEFAULT_MAX_NEIGHBORS
        sim_threshold = sim_threshold or cls.DEFAULT_SIM_THRESHOLD
        min_cluster_size = min_cluster_size or cls.MIN_CLUSTER_SIZE

        pool, support_meta = cls._collect_multihop_pool(
            lemma=lemma,
            vectors_dict=vectors_dict,
            semantic_index=semantic_index,
            max_neighbors=max_neighbors,
            debug=debug
        )

        if not pool:
            if debug:
                print(f"[PUD] {lemma} :: empty pool")
            return []

        G = cls._build_frame_graph(
            lemma=lemma,
            pool=pool,
            vectors_dict=vectors_dict,
            semantic_index=semantic_index,
            support_meta=support_meta,
            sim_threshold=sim_threshold,
            debug=debug
        )

        if G.number_of_nodes() == 0:
            if debug:
                print(f"[PUD] {lemma} :: empty graph")
            return []

        raw_clusters = cls.chinese_whispers(G, seed=42)
        raw_clusters = sorted(raw_clusters, key=len, reverse=True)

        if debug:
            print(f"[PUD] {lemma} :: raw_clusters={[len(c) for c in raw_clusters]}")

        frames = []
        valid_id = 0

        for cluster in raw_clusters:
            if len(cluster) < min_cluster_size:
                continue

            vecs = [vectors_dict[w] for w in cluster if w in vectors_dict]
            if not vecs:
                continue

            sorted_members = sorted(cluster)
            anchors, bridges, background = cls._score_frame_members(cluster, G)
            cohesion = cls._cluster_cohesion(cluster, G)
            frame_type = cls._frame_type(
                members=sorted_members,
                anchors=anchors,
                background_markers=background,
                G=G
            )
            label = cls._frame_label(
                members=sorted_members,
                anchors=anchors,
                background_markers=background,
                G=G,
                frame_type=frame_type
            )

            frames.append({
                "lemma": lemma,
                "sense_id": valid_id,  # kompatybilność
                "frame_id": valid_id,  # NOWE
                "frame_type": frame_type,  # NOWE: semantic / contextual
                "profile_type": frame_type,  # opcjonalny alias przejściowy
                "label": label,
                "members": sorted_members,
                "anchors": anchors[:5],
                "bridge_nodes": bridges[:5],
                "background_markers": background[:5],
                "vector": np.mean(vecs, axis=0),
                "cohesion": cohesion,
                "size": len(sorted_members),

            })
            valid_id += 1

        if not frames and raw_clusters:
            biggest = raw_clusters[0]
            vecs = [vectors_dict[w] for w in biggest if w in vectors_dict]
            if vecs:
                anchors, bridges, background = cls._score_frame_members(biggest, G)
                frame_type = cls._frame_type(
                    members=biggest,
                    anchors=anchors,
                    background_markers=background,
                    G=G
                )
                frames.append({
                    "lemma": lemma,
                    "sense_id": 0,
                    "frame_id": 0,
                    "frame_type": frame_type,
                    "profile_type": frame_type,  # DODAJ TO
                    "label": cls._frame_label(
                        anchors=anchors,
                        background_markers=background,
                        frame_type=frame_type
                    ),
                    "members": sorted(biggest),
                    "anchors": anchors[:5],
                    "bridge_nodes": bridges[:5],
                    "background_markers": background[:5],
                    "vector": np.mean(vecs, axis=0),
                    "cohesion": cls._cluster_cohesion(biggest, G),
                    "size": len(biggest),
                })

        # KLUCZOWE: ramy semantyczne najpierw, dyskursywne na końcu
        frames = sorted(
            frames,
            key=lambda p: (
                0 if p.get("frame_type") == "semantic" else 1,
                -p.get("size", 0),
                -p.get("cohesion", 0.0),
            )
        )

        for i, p in enumerate(frames):
            p["sense_id"] = i
            p["frame_id"] = i

        if debug:
            short = [
                {
                    "id": p["sense_id"],
                    "type": p.get("frame_type"),
                    "size": p["size"],
                    "cohesion": round(p["cohesion"], 3),
                    "label": p["label"],
                    "anchors": p["anchors"][:3],
                    "background": p["background_markers"][:3],
                }
                for p in frames
            ]
            print(f"[PUD] {lemma} :: final_frames={short}")

        return frames
