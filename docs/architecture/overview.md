# Architektura aplikacji

## Obraz całości

`Korpusuj.py` uruchamia GUI. `engine.py` jest rozbudowaną warstwą integracji i orkiestracji GUI: zarządza stanem interfejsu, `app.after`, wątkami i prezentacją. GUI i CLI korzystają ze wspólnego rdzenia `korpusuj/`.

```text
pliki -> creator -> korpus.parquet
                    |-> korpus.search
                    |-> korpus.dep_cache
CQL -> parser -> planner -> backend/SearchCursor -> materializacja -> GUI/CLI
```

## Wspólny rdzeń

Creator GUI i creator CLI wywołują `run_creator_job`. Wyszukiwanie korzysta ze wspólnych modułów parsera, planera, backendu, `SearchCursor` i materializacji. Interfejsy nie utrzymują dwóch niezależnych silników domenowych.

## Dane i subsystemy

Parquet jest kanonicznym magazynem korpusu. `.search` i `.dep_cache` są odtwarzalnymi artefaktami. `korpusuj/semantic/` obsługuje analizy semantyczne, a `korpusuj/topics/` modelowanie BERTopic.

## Katalogi runtime

- `logs/` — logi;
- `models/` — modele NLP;
- `temp/` — zasoby aplikacji i pomoc HTML;
- `fiszki/` — dane fiszek.

## Dalsza lektura

- [Mapa modułów](modules.md)
- [Artefakty](corpus-and-index-artifacts.md)
- [Wyszukiwanie](search-pipeline.md)
- [Creator](creator-pipeline.md)
- [Rozwój](development.md)
