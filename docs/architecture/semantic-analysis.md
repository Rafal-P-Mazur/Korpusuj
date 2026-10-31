## Analiza otoczenia semantycznego

### Dwa workflowy

Pakiet `korpusuj.semantic` udostępnia interaktywny graf eksploracyjny oraz automatyczny raport analityczny. Oba korzystają z przestrzeni FastText, ale nie są dwoma widokami identycznego wyniku.

### Graf eksploracyjny

`SemanticNetworkViewer` przekazuje do `SemanticEngine` lemat początkowy, rozwijany węzeł i lokalny kontekst zbudowanego grafu. Ranking odróżnia bezpośrednie podobieństwo od wyniku kontekstowego używanego do porządkowania kandydatów.

### Graf analityczny

`AnalyticalSemanticReportBuilderV7_1` pobiera pełną dostępną listę sąsiadów i buduje ważony graf mutual 5-NN. Krawędź powstaje, gdy dwa lematy wzajemnie należą do swoich pięciu najbliższych sąsiadów. Wagą jest podobieństwo cosinusowe.

Domyślny kontrakt:

```text
population: full_available_neighbor_list
knn_k: 5
mutual_required: true
edge_weight: cosine
minimum_cluster_size: 3
```

### SenseInducer i Chinese Whispers

Generator raportu wykorzystuje `SenseInducer.chinese_whispers(...)` do podziału przygotowanego grafu. `sense_inducer.py` jest aktywną zależnością raportu, ale nie odpowiada za całość analizy.

```text
reports_analytical_v7_1.py
→ wybór populacji
→ graf mutual 5-NN
→ SenseInducer.chinese_whispers(...)
→ klastry o rozmiarze co najmniej 3
→ centroidy, miary, relacje i eksport
```

Domyślny przebieg używa seedu `42`, najwyżej `100` iteracji i zatrzymania po pełnej iteracji bez zmian etykiet.

### Członkostwo i relacje

`graph_cluster` oznacza członkostwo wynikające z podziału grafu. `frame_relation` opisuje podobieństwo lematu spoza prezentowanych klastrów do dwóch najbliższych centroidów. Relacja centroidowa nie zmienia członkostwa, składu klastra ani centroidu.

### Miary

#### Miary poszczególnych lematów

```text
typicality = cos(v(w), c(r))
distinctiveness = typicality - max cos(v(w), c(s)), s != r
field_typicality = cos(v(w), c(F))
globality = znormalizowany in-degree w zapisanym indeksie sąsiedztwa
field_distinctiveness = field_typicality * (1 - globality)
```

#### Miary poszczególnych ram

```text
size = liczba członków graph_cluster
coverage_share = size / liczba lematów pola
cohesion_centroid_mean = średnia typicality członków ramy
separation_from_other_frames = 1 - max cos(c(r), c(s)), s != r
```

#### Miary całego pola

```text
field_similarity_mean = średnia cos(v(w), v(lemat_centralny))
field_globality_mean = średnia globality
field_cohesion_pairwise = średnia cos(v(w_i), v(w_j)), i < j
frame_cohesion_weighted = średnia cohesion_centroid_mean ważona size
frame_separation_mean = średnia separation_from_other_frames
```

Raport oblicza osobne miary zagregowane dla ram i całego pola, ale nie łączy różnych miar w jeden złożony indeks nośności.
### Eksport

Raport zapisuje `report.html`, kontrakt i diagnostykę JSON, `semantic_field.csv`, `frame_members.csv`, `frame_relations.csv`, `frames.csv`, `frame_similarity.csv` oraz współrzędne projekcji.
