"""BERTopic services for loading documents, chunking text, training or loading models, visualization and topics-over-time analysis."""

import os
import sys
import logging
from pathlib import Path

from korpusuj.runtime_paths import models_root

import pandas as pd


def _default_base_dir_corp() -> str:
    """Resolve the base directory used for topic-modeling corpus resources.
    
    Frozen execution uses the executable directory; source execution resolves the project root from this module path.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    try:
        return str(Path(__file__).resolve().parents[2])
    except Exception:
        return os.getcwd()


class TopicEngine:
    def __init__(self, parquet_path, base_dir_corp=None):
        self.parquet_path = parquet_path
        self.base_dir_corp = base_dir_corp or _default_base_dir_corp()
        self.model = None
        self.topics = None
        self.probs = None
        self.docs = []
        self.timestamps = []

        # Wybór modelu (Sentence Transformer).
        self.embedding_model_name = "sdadas/st-polish-paraphrase-from-mpnet"

    # --- NOWA METODA POMOCNICZA DO CIĘCIA TEKSTÓW ---
    def _chunk_text(self, text, max_words=200):
        """Prosty podział tekstu na mniejsze fragmenty (akapity i okna słów)."""
        chunks = []
        # Podział w pierwszej kolejności na akapity (zachowuje logikę strukturalną)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for p in paragraphs:
            words = p.split()
            if len(words) <= max_words:
                chunks.append(p)
            else:
                # Jeśli sam akapit jest wielki, dzielimy go na sztywne porcje np. po 200 słów
                for i in range(0, len(words), max_words):
                    chunk = " ".join(words[i:i + max_words])
                    chunks.append(chunk)
        return chunks

    # --- ZAKTUALIZOWANA METODA LOAD_DATA ---
    def load_data(self, use_chunking=True, max_words_per_chunk=250, use_lemmas=False): # DODANO PARAMETR
        """Load topic-modeling input through the current corpus processing helpers.
        
        The processing behavior follows :mod:`korpusuj.corpus.creator`; optional
        chunking and lemma selection are controlled by the method arguments.
        """
        logging.info(f"Wczytywanie danych z {self.parquet_path}...")
        try:
            df = pd.read_parquet(self.parquet_path)

            # Wymagane kolumny: Treść i Data publikacji
            if "Treść" not in df.columns:
                raise ValueError("Brak kolumny 'Treść' w pliku parquet.")

            # Odsiewamy puste teksty
            df = df.dropna(subset=['Treść'])
            df = df[df['Treść'].str.strip() != ""]

            # --- NOWE: Wybór między tekstem oryginalnym a lematami ---
            if use_lemmas and "lemmas" in df.columns:
                logging.info("Używam form zlematyzowanych (base) do modelowania BERTopic...")
                # Lematy to listy, więc łączymy je w jeden string " ".join()
                import numpy as np
                raw_docs = df['lemmas'].apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else str(x)).tolist()
            else:
                raw_docs = df['Treść'].tolist()

            # Pobieranie dat, jeśli istnieją
            if "Data publikacji" in df.columns:
                raw_timestamps = df['Data publikacji'].tolist()
            else:
                raw_timestamps = ["0000-00-00"] * len(raw_docs)

            # Resetujemy docelowe listy
            self.docs = []
            self.timestamps = []

            # Aplikujemy chunking
            if use_chunking:
                logging.info("Rozpoczęto chunking długich dokumentów...")
                for doc, ts in zip(raw_docs, raw_timestamps):
                    chunks = self._chunk_text(doc, max_words=max_words_per_chunk)
                    for chunk in chunks:
                        self.docs.append(chunk)
                        self.timestamps.append(ts)  # Kluczowe! Kopiujemy datę dla każdego kawałka
                logging.info(f"Zakończono chunking. Plików orginalnych: {len(raw_docs)}, po podziale: {len(self.docs)}")
            else:
                self.docs = raw_docs
                self.timestamps = raw_timestamps

            logging.info(f"Wczytano {len(self.docs)} dokumentów do analizy tematycznej.")
            return True
        except Exception as e:
            logging.error(f"Błąd ładowania danych: {e}")
            return False

    def train_model(self, nr_topics=None, force_retrain=False, use_stopwords=True, diversity=0.2):
        """Trenuje model BERTopic lub ładuje gotowy z dysku."""
        import os
        from korpusuj.runtime_paths import configure_ml_cache_environment as _configure_ml_cache_environment_182n
        _configure_ml_cache_environment_182n()
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        if not self.docs:
            return False

        model_save_path = self.parquet_path.replace(".parquet", ".bertopic")

        models_dir = models_root() / "sentence-transformers"
        models_dir.mkdir(parents=True, exist_ok=True)
        sentence_model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir))

        if not force_retrain and os.path.exists(model_save_path):
            logging.info(f"Znaleziono gotowy model tematyczny: {model_save_path}. Wczytywanie...")
            self.model = BERTopic.load(model_save_path, embedding_model=sentence_model)
            return True

        # Słownik konfiguracji startowej
        bertopic_config = {
            "embedding_model": sentence_model,
            "language": "polish",
            "calculate_probabilities": False,
            "nr_topics": nr_topics,
            "verbose": True
        }

        # --- Warunkowe podłączenie MMR (Diversity) ---
        if diversity > 0.0:
            from bertopic.representation import MaximalMarginalRelevance
            representation_model = MaximalMarginalRelevance(diversity=diversity)
            bertopic_config["representation_model"] = representation_model
            logging.info(f"Włączono algorytm MMR wymuszający różnorodność słów (diversity={diversity})")



        # --- Warunkowe podłączenie stoplisty ---
        if use_stopwords:
            polish_stopwords = [
                "a", "aby", "ach", "acz", "aczkolwiek", "aj", "albo", "ale", "alez", "ależ", "ani", "az", "aż",
                "bardziej", "bardzo", "beda", "bedzie", "bez", "deda", "będą", "bede", "będę", "będzie", "bo",
                "bowiem", "by", "byc", "być", "byl", "byla", "byli", "bylo", "byly", "był", "była", "było",
                "były", "bynajmniej", "cala", "cali", "caly", "cała", "cały", "ci", "cie", "ciebie", "cię", "co",
                "cokolwiek", "cos", "coś", "czasami", "czasem", "czemu", "czy", "czyli", "daleko", "dla",
                "dlaczego", "dlatego", "do", "dobrze", "dokad", "dokąd", "dosc", "dość", "duzo", "dużo", "dwa",
                "dwaj", "dwie", "dwoje", "dzis", "dzisiaj", "dziś", "gdy", "gdyby", "gdyz", "gdyż", "gdzie",
                "gdziekolwiek", "gdzies", "gdzieś", "go", "i", "ich", "ile", "im", "inna", "inne", "inny",
                "innych", "iz", "iż", "ja", "jak", "jakas", "jakaś", "jakby", "jaki", "jakichs", "jakichś",
                "jakie", "jakis", "jakiś", "jakiz", "jakiż", "jakkolwiek", "jako", "jakos", "jakoś", "ją", "je",
                "jeden", "jedna", "jednak", "jednakze", "jednakże", "jedno", "jego", "jej", "jemu", "jesli",
                "jest", "jestem", "jeszcze", "jeśli", "jezeli", "jeżeli", "juz", "już", "kazdy", "każdy", "kiedy",
                "kilka", "kims", "kimś", "kto", "ktokolwiek", "ktora", "ktore", "ktorego", "ktorej", "ktory",
                "ktorych", "ktorym", "ktorzy", "ktos", "ktoś", "która", "które", "którego", "której", "który",
                "których", "którym", "którzy", "ku", "lat", "lecz", "lub", "ma", "mają", "mało", "mam", "mi",
                "miedzy", "między", "mimo", "mna", "mną", "mnie", "moga", "mogą", "moi", "moim", "moj", "moja",
                "moje", "moze", "mozliwe", "mozna", "może", "możliwe", "można", "mój", "mu", "musi", "my", "na",
                "nad", "nam", "nami", "nas", "nasi", "nasz", "nasza", "nasze", "naszego", "naszych", "natomiast",
                "natychmiast", "nawet", "nia", "nią", "nic", "nich", "nie", "niech", "niego", "niej", "niemu",
                "nigdy", "nim", "nimi", "niz", "niż", "no", "o", "obok", "od", "około", "on", "ona", "one",
                "oni", "ono", "oraz", "oto", "owszem", "pan", "pana", "pani", "po", "pod", "podczas", "pomimo",
                "ponad", "poniewaz", "ponieważ", "powinien", "powinna", "powinni", "powinno", "poza", "prawie",
                "przeciez", "przecież", "przed", "przede", "przedtem", "przez", "przy", "roku", "rowniez",
                "również", "sam", "sama", "są", "sie", "się", "skad", "skąd", "soba", "sobą", "sobie", "sposob",
                "sposób", "swoje", "ta", "tak", "taka", "taki", "takie", "takze", "także", "tam", "te", "tego",
                "tej", "ten", "teraz", "też", "to", "toba", "tobą", "tobie", "totez", "toteż", "tobą", "trzeba",
                "tu", "tutaj", "twoi", "twoim", "twoj", "twoja", "twoje", "twój", "twym", "ty", "tych", "tylko",
                "tym", "u", "w", "wam", "wami", "was", "wasz", "wasza", "wasze", "we", "według", "wiele", "wielu",
                "więc", "więcej", "wlasnie", "właśnie", "wszyscy", "wszystkich", "wszystkie", "wszystkim",
                "wszystko", "wtedy", "wy", "z", "za", "zaden", "zadna", "zadne", "zadnych", "zapewne", "zawsze",
                "ze", "zeby", "znowu", "zł", "znow", "znowu", "znów", "zostal", "został", "żaden", "żadna",
                "żadne", "żadnych", "że", "żeby"
            ]
            vectorizer_model = CountVectorizer(stop_words=polish_stopwords)
            bertopic_config["vectorizer_model"] = vectorizer_model
            logging.info("Dołączono własną listę stop-words do wektoryzatora.")

        logging.info(f"Rozpoczynam trening od zera z parametrem nr_topics={nr_topics}...")

        # KORPUSUJ_PATCH_180Y_BERTOPIC_HDBSCAN_SINGLE_PROCESS
        # BERTopic 0.17.4 normally constructs this HDBSCAN model internally.
        # Keep its clustering contract unchanged, but prevent joblib/loky from
        # spawning four frozen Korpusuj worker processes for core distances.
        bertopic_config["hdbscan_model"] = HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=1,
        )

        self.model = BERTopic(**bertopic_config)
        self.topics, self.probs = self.model.fit_transform(self.docs)

        logging.info(f"Trening zakończony. Zapisuję model do: {model_save_path}")
        self.model.save(model_save_path)

        return True

    def get_topic_info(self):
        """Zwraca DataFrame z informacjami o tematach (ID, Liczba tekstów, Słowa kluczowe)."""
        if self.model:
            return self.model.get_topic_info()
        return None

    def calculate_topics_over_time(self):
        """Generuje trendy bez ryzyka asynchronizacji danych po wczytaniu z cache."""
        if not self.model:
            return None

        # Przekazujemy absolutnie wszystkie dokumenty i daty, bez wycinania 0000-00-00.
        # Puste daty pojawią się po prostu jako pierwszy punkt na lewo od wykresu.
        try:
            topics_over_time = self.model.topics_over_time(
                self.docs,
                self.timestamps,
                nr_bins=20
            )
            return topics_over_time
        except Exception as e:
            logging.info(f"Błąd podczas obliczania trendów w czasie: {e}")
            return None

    def visualize_dynamic_topics(self, topics_over_time, top_n_topics=15):
        """Zwraca interaktywny wykres Plotly obrazujący trendy w czasie."""
        if self.model and topics_over_time is not None:
            return self.model.visualize_topics_over_time(topics_over_time, top_n_topics=top_n_topics)
        return None

    def visualize_topic_map(self):
        """Return an optional intertopic UMAP map when the model has enough topics.

        BERTopic delegates this visualization to UMAP. For tiny corpora one
        non-outlier topic may be a valid training result, but it cannot form an
        intertopic graph. A missing map must not invalidate the saved model.
        """
        if not self.model:
            return None

        try:
            topic_info = self.model.get_topic_info()
            if topic_info is not None and "Topic" in topic_info.columns:
                valid_topic_ids = {
                    int(topic_id)
                    for topic_id in topic_info["Topic"].tolist()
                    if int(topic_id) != -1
                }
                if len(valid_topic_ids) < 2:
                    logging.warning(
                        "Mapa tematów pominięta: potrzeba co najmniej 2 tematów innych niż -1, znaleziono %s.",
                        len(valid_topic_ids),
                    )
                    return None
        except Exception as exc:
            logging.warning(
                "Nie udało się ustalić liczby tematów przed wizualizacją; podejmuję próbę utworzenia mapy: %s",
                exc,
            )

        try:
            return self.model.visualize_topics()
        except Exception as exc:
            logging.warning(
                "Mapa tematów jest niedostępna; model pozostaje poprawnie wytrenowany i zapisany: %s",
                exc,
            )
            return None

    def visualize_word_scores(self, top_n_topics=10):
        """Generuje wykres słupkowy Word scores (c-TF-IDF) dla top tematów."""
        if self.model:
            # Pokazuje najważniejsze słowa i ich wagi dla wybranych tematów
            return self.model.visualize_barchart(top_n_topics=top_n_topics, n_words=10)
        return None
