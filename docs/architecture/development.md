# Rozwój i testowanie

## Praca z kodem

Przed zmianą danego obszaru sprawdź jego publiczny interfejs i odpowiadające mu testy. Zacznij od modułu będącego właścicielem funkcji, a następnie prześledź miejsca, które wywołują ten kod z GUI lub CLI.

Część komentarzy w kodzie opisuje wcześniejsze etapy rozwoju aplikacji. Przy ustalaniu bieżącego zachowania kieruj się aktualną implementacją, publicznym interfejsem i uruchamianymi testami.

## Testy

Zestaw obejmuje 152 testy w 17 plikach: creator 58, search 64, coref 26 i GUI 4.

```powershell
python -m pytest -q -p no:cacheprovider tests
```

## Własność kodu

Creator należy do `korpusuj/corpus/`, `.search` do `korpusuj/index/`, dependency do `korpusuj/dependency/`, wyszukiwanie do `korpusuj/search/`, eksport do `korpusuj/export/`, a GUI do `engine.py` i `korpusuj/ui/`.

Nie należy dodawać helpera ani klasy bez uproszczenia kodu lub rzeczywistej wspólnej granicy.

## Workspace i runtime

Skanery, generatory i patche nie mogą używać systemowego `%TEMP%`. Workspace ma znajdować się w rootcie projektu.

- `logs/gui/korpusuj.log` — log GUI;
- `logs/search_cli/` — diagnostyka CLI;
- `models/` — modele;
- `temp/` — zasoby aplikacji;
- `fiszki/` — dane fiszek.

## `engine.py`

`engine.py` jest rozbudowaną warstwą GUI z istotnym stanem, schedulingiem, wątkami i prezentacją. Nowa semantyka domenowa powinna trafiać do odpowiedniego pakietu.

## Aktualizowanie dokumentacji

Po zmianie publicznego interfejsu, formatu danych albo przebiegu przetwarzania zaktualizuj odpowiadającą mu stronę dokumentacji. Przykłady poleceń powinny działać z bieżącą wersją CLI, a opisy przepływów powinny odpowiadać zachowaniu objętemu testami.

Jeżeli zmiana dotyczy kilku warstw, zaktualizuj najpierw stronę opisującą cały przepływ, a następnie mapę modułów. Dzięki temu dokumentacja pozostaje użyteczna zarówno dla osoby poznającej projekt, jak i dla osoby szukającej właściciela konkretnej funkcji.

