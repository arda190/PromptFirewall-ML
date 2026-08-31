from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TFIDFVectorizer:
    def __init__(self,max_features=None,min_df=1, max_df=1.0,ngram_range=(3,5),**kwargs):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            **kwargs
        )
        self.name = "TF-IDF"

    def fit(self, sentences):
        self.vectorizer.fit(sentences)
        return self

    def transform(self, sentences):
        return self.vectorizer.transform(sentences)

    def fit_transform(self, sentences):
        return self.vectorizer.fit_transform(sentences)

    def save_model(self, filename):
        joblib.dump(self.vectorizer, filename)

    def load_model(self, filename="saved_models/tf-idf"):
        self.vectorizer = joblib.load(filename)
        return self