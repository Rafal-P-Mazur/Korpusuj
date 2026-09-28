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

```text
typicality = cos(v(w), c(r))
distinctiveness = typicality - max cos(v(w), c(s)), s != r
field_typicality = cos(v(w), c(F))
field_distinctiveness = field_typicality * (1 - globality)
```

Raport nie tworzy agregowanego indeksu nośności.

### Eksport

Raport zapisuje `report.html`, kontrakt i diagnostykę JSON, `semantic_field.csv`, `frame_members.csv`, `frame_relations.csv`, `frames.csv`, `frame_similarity.csv` oraz współrzędne projekcji.
