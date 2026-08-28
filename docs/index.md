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

## Architektura i rozwój

Dokumenty w tej części są przeznaczone dla osób rozwijających i utrzymujących aplikację.

- [Architektura aplikacji](architecture/overview.md)
- [Mapa modułów](architecture/modules.md)
- [Korpus i artefakty indeksowe](architecture/corpus-and-index-artifacts.md)
- [Przepływ wyszukiwania](architecture/search-pipeline.md)
- [Przepływ tworzenia korpusu](architecture/creator-pipeline.md)
- [Rozwój i testowanie](architecture/development.md)
