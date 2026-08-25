import pandas as pd
from sklearn.model_selection import train_test_split

from experiment.experiment import Experiment
from pipeline.Pipeline import Pipeline
from evaluation.Evaluator import Evaluator

from vectorizers.BagOfWords import BagOfWords
from vectorizers.TFIDF import TFIDF
from vectorizers.Word2Vec import Word2Vectorizer
from vectorizers.FastText import FastTextVectorizer

from models.from_scratch.NaiveBayes import NaiveBayes
from models.library.SupportVectorMachine import SupportVectorMachine


def load_dataset():
    data = pd.read_csv("datasets/generated_2.csv")

    X = data["text"].tolist()
    y = data["label"].tolist()

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )


def create_experiments():
    return [

        # -------------------------
        # Bag of Words
        # -------------------------

        Experiment(
            Pipeline(BagOfWords(), NaiveBayes()),
            Evaluator(),
            "bow_naivebayes.json"
        ),

        Experiment(
            Pipeline(BagOfWords(), SupportVectorMachine()),
            Evaluator(),
            "bow_svm.json"
        ),

        # -------------------------
        # TF-IDF
        # -------------------------

        Experiment(
            Pipeline(TFIDF(), NaiveBayes()),
            Evaluator(),
            "tfidf_naivebayes.json"
        ),

        Experiment(
            Pipeline(TFIDF(), SupportVectorMachine()),
            Evaluator(),
            "tfidf_svm.json"
        ),

        # -------------------------
        # Word2Vec
        # -------------------------

        Experiment(
            Pipeline(Word2Vectorizer(), SupportVectorMachine()),
            Evaluator(),
            "word2vec_svm.json"
        ),

        # -------------------------
        # FastText
        # -------------------------

        Experiment(
            Pipeline(FastTextVectorizer(), SupportVectorMachine()),
            Evaluator(),
            "fasttext_svm.json"
        ),
    ]


def main():

    X_train, X_test, y_train, y_test = load_dataset()

    experiments = create_experiments()

    print("=" * 60)
    print("PromptFirewall-ML")
    print("Running Experiments...")
    print("=" * 60)

    for experiment in experiments:
        print("=" * 60)
        print(f"Running: {experiment.filename}")

        result = experiment.run(
            X_train,
            y_train,
            X_test,
            y_test
        )

        experiment.save_experiment()

        metrics = result["metrics"]

        print(f"Vectorizer : {type(experiment.pipeline.vectorizer).__name__}")
        print(f"Classifier : {type(experiment.pipeline.classifier).__name__}")
        print(f"Accuracy   : {metrics['accuracy']:.4f}")
        print(f"Precision  : {metrics['precision']:.4f}")
        print(f"Recall     : {metrics['recall']:.4f}")
        print(f"F1-score   : {metrics['f1_score']:.4f}")

    print("=" * 60)
    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()