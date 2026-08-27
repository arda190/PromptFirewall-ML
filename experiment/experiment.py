import json
import os
import time
import numpy as np
import time

class Experiment:
    def __init__(self,pipeline,evaluator,filename):
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.filename = filename
        self.metrics = None
        self.test_time = None
        self.train_size = None
        self.test_size = None



    def run(self,X_train,X_test, y_test):

        predictions = []

        start = time.perf_counter()

        for text in X_test:
            prediction = self.pipeline.predict(text)

            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()

            predictions.append(prediction)

        end = time.perf_counter()

        self.test_time = end - start

        results = self.evaluator.evaluate(y_test, predictions)

        self.metrics = results

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
            "test_time": f"{self.test_time:.4f} seconds for {self.test_size} samples",
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

