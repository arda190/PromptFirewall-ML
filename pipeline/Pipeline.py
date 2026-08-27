from preprocessing.Cleaner import Cleaner
import os
from vectorizers.library.factory import get_vectorizer
from models.library.model_factory import get_model

class Pipeline:
    def __init__(self,vectorizer_path,classifier_path): # path is not an absolute path ,  factory classes will add saved_models/
        self.cleaner = Cleaner()
        self.vectorizer = get_vectorizer(vectorizer_path)
        self.classifier = get_model(classifier_path)




    def predict(self,input):
        input = self.cleaner.clean_text(input)
        X = self.vectorizer.transform([input])

        return self.classifier.predict(X)
