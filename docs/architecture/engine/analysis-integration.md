## Integracja analiz w engine.py

### Sieć semantyczna i raport analityczny

`semantic_engine = SemanticEngine()` powstaje podczas importu `engine.py`. Graf eksploracyjny korzysta z metod `SemanticEngine` do pobierania i kontekstowego porządkowania kandydatów. Raport jest uruchamiany przez `SemanticEngine.build_semantic_report(...)` i generowany przez `reports_analytical_v7_1.py`.

Generator raportu buduje graf mutual k-NN, a do jego podziału wykorzystuje `SenseInducer.chinese_whispers(...)`. Następnie oblicza centroidy, miary ramowe i polowe, relacje lematów spoza klastrów oraz podobieństwa między centroidami.

### Profil słowa

`compute_word_profile` i `flatten_word_profile` obsługują profil kolokacyjny. Jest to analiza odrębna od grafu eksploracyjnego i raportu semantycznego.

### BERTopic

`TopicEngine` obsługuje analizę tematyczną i wygenerowany raport.

### Raporty HTML i fiszki

`launch_webview(target_path)` otwiera gotowy raport HTML. `get_fiszki_module()` importuje moduł fiszek na żądanie.
