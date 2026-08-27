import os
from models.library import *
from models.library.LightGBM import LightGBMWrapper
from models.library.LogisticRegression import LogisticRegressionWrapper
from models.library.NaiveBayes import NaiveBayesWrapper
from models.library.SupportVectorMachine import SupportVectorMachineWrapper


def get_model(model_name):
    model = model_name.split("_")[1]
    print(model)

    path = os.path.join("saved_models", model_name)

    match model:
        case "logisticregression":
            vectorizer = LogisticRegressionWrapper()
            vectorizer = vectorizer.load_model(path)
            return vectorizer

        case "SupportVectorMachine":
            vectorizer = SupportVectorMachineWrapper()
            vectorizer = vectorizer.load_model(path)
            return vectorizer

        case "NaiveBayes":
            vectorizer = NaiveBayesWrapper()
            vectorizer = vectorizer.load_model(path)
            return vectorizer
        case "LightGBM":
             classifier = LightGBMWrapper()
             classifier = classifier.load_model(path)
             return classifier

        case _:
            raise ValueError(f"Unknown model: {model_name}")