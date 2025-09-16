# Korpusuj

Aplikacja do tworzenia i przeszukiwania automatycznie anotowanych korpusów tekstów.

![Tag](https://img.shields.io/github/v/tag/Rafal-P-Mazur/Korpusuj)
---

## Funkcje
- Tworzenie automatycznie anotowanych korpusów tekstów (z wykorzystaniem biblioteki Stanza).
- Przeszukiwanie i filtrowanie danych tekstowych.
- Generowanie i wizualizacja danych frekwencyjnych.
- Gotowy plik wykonywalny (.exe) dla Windows – nie wymaga instalacji Pythona.

---

## Instalacja i użytkowanie

### Windows (.exe)
1. Pobierz najnowszą wersję z [Releases](https://github.com/Rafal-P-Mazur/Korpusuj/releases).
2. Rozpakuj archiwum ZIP.
3. Uruchom `Korpusuj.exe` klikając dwukrotnie.

⚠️ Plik nie jest podpisany certyfikatem Windows – system może wyświetlić ostrzeżenie bezpieczeństwa. Jest to normalne.
ℹ️ Plik `.exe` zawiera wersję biblioteki **Torch** skonfigurowaną do pracy **tylko na CPU**.  

---

### Python
1. Zainstaluj Python 3.10.
2. Sklonuj repozytorium:
   ```bash
   git clone [https://github.com/Rafal-P-Mazur/REPO.git](https://github.com/Rafal-P-Mazur/Korpusuj.git)
3. Zainstaluj wymagania:
  pip install -r requirements.txt
  ⚠️Aby podczas tworzenia korpusu wykorzystywać GPU, należy pobrać wersję Torch zgodną z kartą graficzną)
4. Uruchom aplikajcę python Korpusuj_beta.py
