import json
import os
import time
import numpy as np

class Experiment:
    def __init__(self,pipeline,evaluator,filename):
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.filename = filename
        self.metrics = None
        self.predictions = None
        self.trainig_time = None
        self.train_size = None
        self.test_size = None



    def run(self, X_train, y_train, X_test, y_test):

        start = time.time()
        self.pipeline.fit(X_train, y_train)
        end = time.time()

        self.trainig_time = end - start

        predictions = []

        for text in X_test:
            prediction = self.pipeline.predict(text)

            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()

            predictions.append(prediction)


        results = self.evaluator.evaluate(y_test, predictions)

        self.metrics = results
        self.predictions = predictions

        self.train_size = len(X_train)
        self.test_size =len(X_test)


        return {
            "metrics": results,
            "predictions": predictions
        }

    def save_experiment(self):
        os.makedirs("experiment_results", exist_ok=True)
        filepath = os.path.join("experiment_results", self.filename)

        data ={
            "vectorizer": type(self.pipeline.vectorizer).__name__,
            "classifier": type(self.pipeline.classifier).__name__,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metrics": self.metrics,
            "predictions": self.predictions
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

