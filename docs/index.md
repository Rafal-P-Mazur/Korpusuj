# Dokumentacja Korpusuj

Dokumentacja obejmuje instalację, pierwsze uruchomienie, obsługę interfejsu graficznego i wiersza poleceń, język zapytań CQL oraz informacje techniczne przeznaczone dla osób rozwijających aplikację.

## Zacznij tutaj

- [Instalacja](installation.md) — przygotowanie środowiska, instalacja zależności i uruchomienie aplikacji.
- [Pierwsze kroki](quickstart.md) — otwarcie lub utworzenie korpusu i wykonanie pierwszego wyszukiwania.

## Instrukcje użytkownika

- [Interfejs graficzny](gui.md) — tworzenie i otwieranie korpusów, wyszukiwanie, statystyki, wykresy, kolokacje, sieć semantyczna, modelowanie tematyczne i eksport danych.
- [Interfejs wiersza poleceń](cli.md) — tworzenie korpusów, zarządzanie indeksami, wyszukiwanie, analizy i eksport z terminala.
- [Język zapytań CQL](cql.md) — składnia zapytań od podstawowych warunków tokenowych po relacje składniowe, NER, koreferencję, filtry zdań i metadane.
- [Scalanie gotowych korpusów](corpus_merger.md)
- [Korekta lematyzacji gotowego korpusu](lemma-repair.md) — audyt SGJP, bezpieczne reguły, walidacja i artefakty.

### Architektura

- [Spis dokumentacji architektury](architecture/index.md)
- [Obraz systemu](architecture/overview.md)
- [Uruchamianie i stan aplikacji](architecture/application-runtime.md)
- [Format korpusu](architecture/corpus-format.md)
- [Otwieranie korpusu](architecture/corpus-loading.md)
- [Tworzenie korpusu](architecture/creator-pipeline.md)
- [Indeks wyszukiwania i pamięć zależnościowa](architecture/index-lifecycle.md)
- [Wykonanie zapytania](architecture/search-execution.md)
- [Dostęp do relacji zależnościowych](architecture/dependency-runtime.md)
- [Wyniki, liczba trafień i eksport](architecture/result-materialization.md)
- [Połączenie modułów z GUI](architecture/gui-integration.md)
- [Analiza otoczenia semantycznego](architecture/semantic-analysis.md)
- [Mapa pakietów i ważnych modułów](architecture/modules.md)
- [Referencja modułów i symboli](architecture/source-reference.md)
