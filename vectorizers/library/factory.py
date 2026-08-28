import os
from pathlib import Path
from data_loader import load_dataset
from vectorizers.library.FastText import FastTextVectorizer
from vectorizers.library.TF_IDF import TFIDFVectorizer
from vectorizers.library.Word2Vec import Word2Vectorizer


def load_vectorizer(vectorizer_name):
    path = os.path.join("saved_models", vectorizer_name)

    match vectorizer_name:
        case "tf-idf":
            vectorizer = TFIDFVectorizer()
            vectorizer = vectorizer.load_model(path)
            return vectorizer

        case "fasttext":
            vectorizer = FastTextVectorizer()
            vectorizer = vectorizer.load_model(path)
            return vectorizer

        case "word2vec":
            vectorizer = Word2Vectorizer()
            vectorizer = vectorizer.load_model(path)
            return vectorizer
        case _:
            raise ValueError("Invalid vectorizer name")

def train_vectorizer(vectorizer_name):
    path = os.path.join("saved_models", vectorizer_name)
    X_train = load_dataset()[0]

    match vectorizer_name:
        case "tf-idf":
            vectorizer = TFIDFVectorizer()
            vectorizer.fit(X_train)
            vectorizer.save_model(path)
            return vectorizer

        case "fasttext":
            vectorizer = FastTextVectorizer()
            vectorizer.fit(X_train)
            vectorizer.save_model(path)
            return vectorizer

        case "word2vec":
            vectorizer = Word2Vectorizer()
            vectorizer.fit(X_train)
            vectorizer.save_model(path)
            return vectorizer

        case _:
            raise ValueError("Invalid vectorizer name")


def get_vectorizer(vectorizer_name):
    path = os.path.join("saved_models", vectorizer_name)
    path = Path(path)

    if path.exists():
        return load_vectorizer(vectorizer_name)
    else:
        return train_vectorizer(vectorizer_name)
