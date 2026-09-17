# Uruchamianie `engine.py`

## Kod wykonywany przed utworzeniem GUI

`engine.py` najpierw rozpoznaje argumenty przeznaczone dla pomocniczych trybów procesu. Obsługuje między innymi otwarcie raportu HTML w webview, zadania semantyczne i wejście fiszek. Jeśli taki argument jest obecny, odpowiednia gałąź kończy proces przez `sys.exit(...)` przed utworzeniem głównego okna.

Następnie moduł ustawia kodowanie strumieni, importuje biblioteki GUI i moduły aplikacji, konfiguruje logowanie oraz ładuje ustawienia z lokalizacji wyznaczonej przez `runtime_paths`.

## Importy i inicjalizacja wykonywane globalnie

W czasie importu tworzone są lub konfigurowane elementy używane przez późniejsze funkcje GUI:

- `TopicEngine` i integracja analiz tematycznych;
- obiekt `SemanticEngine` przypisany do `semantic_engine`;
- wrappery parsera CQL delegujące do `korpusuj.search.parser`;
- konfiguracja `SearchExecutor` klasami `SearchCursor` i `SearchIndex`;
- `DependencyRuntimeState` oraz bindingi odczytujące stan z `engine.py`;
- konfiguracja `SearchCursor` funkcjami dependency i dostawcami ustawień.

To są skutki importu modułu. `main()` nie tworzy tych elementów ponownie.

## `main()`

`main()` buduje główne okno i widoki, wiąże callbacki oraz uruchamia pętlę zdarzeń. Znaczna część kontrolek i zmiennych stanu jest tworzona w module przed albo podczas wykonania `main()`.

## `on_closing()`

Callback zamknięcia kończy pracę GUI i zwalnia zasoby związane z otwartymi korpusami oraz zadaniami w tle. Jest podłączony do zdarzenia zamknięcia głównego okna.

## Leniwe importy funkcji

`get_creator_module()` i `get_fiszki_module()` importują cięższe części aplikacji dopiero przed pierwszym użyciem. Zwrócony moduł jest buforowany, więc następne otwarcie funkcji nie wykonuje ponownie pełnego importu.
