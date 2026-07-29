from preprocessing.Cleaner import Cleaner
import os


class Pipeline:
    def __init__(self,vectorizer,classifier):
        self.cleaner = Cleaner()
        self.vectorizer = vectorizer
        self.classifier = classifier



    def fit(self,X_train,Y_train):
        X_train = self.cleaner.clean(X_train)
        X = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X,Y_train)


    def predict(self,input):
        input = self.cleaner.clean_text(input)
        X = self.vectorizer.transform([input])

        return self.classifier.predict(X)


    def save_model(self,classifier_name,vectorizer_name):
        classifier_path = os.path.join("saved_models", classifier_name)
        vectorizer_path = os.path.join("saved_models", vectorizer_name)
        self.classifier.save_model(classifier_path)
        self.vectorizer.save_model(vectorizer_path)

    def load_model(self,classifier_name,vectorizer_name):
        classifier_path = os.path.join("saved_models", classifier_name)
        vectorizer_path = os.path.join("saved_models", vectorizer_name)
        self.classifier.load_model(classifier_path)
        self.vectorizer.load_model(vectorizer_path)