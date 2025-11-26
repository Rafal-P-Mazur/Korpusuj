# Korpusuj

Aplikacja do tworzenia i przeszukiwania automatycznie anotowanych korpusów tekstów.

![Tag](https://img.shields.io/github/v/tag/Rafal-P-Mazur/Korpusuj)
---

![image alt](https://github.com/Rafal-P-Mazur/Korpusuj/blob/main/Images/1.png?raw=true)

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
   git clone https://github.com/Rafal-P-Mazur/Korpusuj.git
3. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt

   ⚠️ Aby podczas tworzenia korpusu wykorzystywać GPU, należy pobrać zgodną z kartą graficzną wersję Torch

4. Uruchom aplikację:
   ```bash
   python Korpusuj_beta.py

