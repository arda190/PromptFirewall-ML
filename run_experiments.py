import pandas as pd
from gensim.models import FastText
from sklearn.model_selection import train_test_split

from experiment.experiment import Experiment
from pipeline.Pipeline import Pipeline
from evaluation.Evaluator import Evaluator

from vectorizers.Custom.BagOfWords import BagOfWords
from vectorizers.Custom.TFIDF import TFIDF
from vectorizers.library.Word2Vec import Word2Vectorizer
from vectorizers.library.FastText import FastTextVectorizer

from models.library.NaiveBayes import NaiveBayesWrapper
from models.library.SupportVectorMachine import SupportVectorMachineWrapper
from vectorizers.library.TF_IDF import TFIDFVectorizer

from models.library.LightGBM import  LightGBMWrapper

from models.library.LogisticRegression import  LogisticRegressionWrapper
from data_loader import load_dataset





def create_experiments():
    return [
        Experiment(
            Pipeline("tf-idf","tf-idf_SupportVectorMachine"),
            Evaluator(),
            "tf-idf_svm(2).json"
        ),

        Experiment(
            Pipeline("tf-idf", "tf-idf_LightGBM"),
            Evaluator(),
            "tf-idf_LightGBM(2).json"
        ),

        Experiment(
            Pipeline("word2vec","word2vec_SupportVectorMachine"),
            Evaluator(),
            "word2vec_svm(2).json"
        ),

        Experiment(
            Pipeline("word2vec", "word2vec_logisticregression"),
            Evaluator(),
            "word2vec_logisticRegression(2).json"
        ),

        Experiment(
            Pipeline("word2vec", "word2vec_LightGBM"),
            Evaluator(),
            "word2vec_LightGBM(2).json"
        ),

        Experiment(
            Pipeline("fasttext","fasttext_LightGBM"),
            Evaluator(),
            "fasttext_LightGBM(2).json"
        )

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

        result = experiment.run(  # experiment run sadece test verilerini test edecek . Model eğitimi yok
            X_train,
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