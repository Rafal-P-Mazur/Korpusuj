## Lematyzacja

### Lematyzacja i korekta lematyzacji

Korpusuj rozróżnia:

- **lematyzator tworzący korpus** — procesor pipeline'u NLP, który podczas anotacji zapisuje pierwotną kolumnę `lemmas`;
- **korektę lematyzacji gotowego korpusu** — późniejszy, opcjonalny przebieg wykorzystujący istniejące tokeny, tagi, NER oraz Morfeusza 2/SGJP i zapisujący oddzielny plik Parquet.

Wyniku „Lematyzator Korpusuj z korektą lematyzacji” nie należy interpretować jako wyniku samego modelu. Wartość opisuje dwa kolejno zastosowane etapy.

### Lematyzator Korpusuj

**Lematyzator Korpusuj** jest modelem używanym wyłącznie jako procesor `lemma` w pipeline Stanza. Nie zmienia tokenizacji, MWT, POS/XPOS/FEATS, NER, parsera zależnościowego ani koreferencji.

W GUI po wybraniu Stanza dostępne są:

- **Lematyzator Korpusuj** — wariant domyślny;
- **Domyślny lematyzator Stanza** — wariant zgodnościowy i porównawczy.

W CLI odpowiadają im `--stanza-lemmatizer korpusuj` oraz `--stanza-lemmatizer stanza`. Parametr nie dotyczy spaCy.

### Dystrybucja i pobieranie modeli

Model Lematyzatora Korpusuj jest pakowany z aplikacją jako `temp/lemmatizer_korpusuj.pt`.

Pozostałe modele Stanza i spaCy nie zmieniają swojej polityki dystrybucji. Przy pierwszym tworzeniu korpusu kreator sprawdza katalog modeli i pobiera brakujące składniki. Wybór Lematyzatora Korpusuj nie eliminuje potrzeby posiadania pozostałych procesorów Stanza.

### Ewaluacja end-to-end na PDB

Ewaluację wykonano na `pl_pdb-ud-test.conllu`, od odtworzonego surowego tekstu do pełnego systemowego CoNLL-U. Wszystkie warianty korzystały z identycznej tokenizacji, tagowania i parsowania. Wariant Lematyzatora Korpusuj różnił się lematyzatorem, a wariant skorygowany dodatkowo stosował korektę lematyzacji. Metryki policzono oficjalnym `conll18_ud_eval.py`.

| Wariant | Lemmas F1 | Lemmas aligned accuracy | BLEX F1 |
|---|---:|---:|---:|
| Lematyzator Korpusuj | **97,74** | **97,88** | **85,71** |
| Lematyzator Korpusuj z korektą lematyzacji | **98,08** | **98,23** | **86,18** |

### Korekta lematyzacji

Korekta lematyzacji jest dostępna dla gotowego korpusu przez **Plik → Popraw lematyzację aktywnego korpusu**. Działa konserwatywnie i:

- nie zmienia tekstu ani tokenizacji;
- nie zmienia tagów, zależności, NER ani koreferencji;
- korzysta z Morfeusza 2/SGJP, pełnego XPOS i sygnałów NER;
- blokuje homografię oraz cele niejednoznaczne;
- zawiera ochronę skrótów i nazw własnych;
- obsługuje bezpieczne przypadki kapitalizacji początku zdania i skrótowców przed łącznikiem;
- zapisuje nowy plik Parquet, pozostawiając źródło bez zmian.

Szczegółowy workflow, artefakty i bramki publikacji opisuje dokument [Korekta lematyzacji gotowego korpusu](lemma-repair.md).

### Reprodukowalność

Przy opisie korpusu należy zapisać co najmniej:

- backend NLP (`stanza` albo `spacy`);
- dla Stanza: wybrany lematyzator (`korpusuj` albo `stanza`);
- stan warstw NER i koreferencji;
- informację, czy zastosowano korektę lematyzacji;
- wersję aplikacji lub identyfikator wersji modelu, jeśli korpus ma służyć porównaniom eksperymentalnym.
