# Integracja analiz w `engine.py`

## Sieć semantyczna

`semantic_engine = SemanticEngine()` powstaje podczas importu `engine.py`. Funkcje `load_semantic_neighbors(...)`, `get_semantic_neighbors(...)`, `is_mutual_knn(...)` i `dynamic_bridge_threshold(...)` delegują do tego obiektu albo do metod klasy.

`smart_show_semantic_network()` sprawdza dane wybranego korpusu i otwiera widok sieci. Po zakończeniu treningu `on_training_success()` aktualizuje status i ponawia wczytanie artefaktów.

## Profil słowa i indukcja sensów

`compute_word_profile`, `flatten_word_profile` i `SenseInducer` są importowane przez `engine.py` i używane przez callbacki panelu semantycznego. Parametry pochodzą z kontrolek GUI, a wynik jest przekazywany do widoku lub generatora raportu.

## BERTopic

`TopicEngine` jest importowany podczas inicjalizacji modułu. Callback analizy tematycznej zbiera ustawienia, uruchamia obliczenia poza bieżącą obsługą zdarzenia i otwiera wygenerowany raport.

## Raporty HTML

`launch_webview(target_path)` uruchamia pywebview dla gotowego pliku HTML. W wersji zamrożonej osobne zadanie procesu jest rozpoznawane na początku `engine.py`, zanim zostanie utworzone główne GUI.

## Fiszki

`get_fiszki_module()` importuje moduł fiszek na żądanie. Osobny argument `--run-fiszki` pozwala uruchomić wejście fiszek w procesie pomocniczym.
