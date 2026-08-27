from sklearn.linear_model import LogisticRegression
import joblib


class LogisticRegressionWrapper:

    def __init__(self,C=1.0,max_iter=1000,solver="lbfgs",penalty="l2",class_weight=None,random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.penalty = penalty
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            penalty=penalty,
            class_weight=class_weight,
            random_state=random_state
        )
        self.name = "LogisticRegression"

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self,path):
        joblib.dump(self.model,path)

    def load_model(self,path):
        self.model = joblib.load(path)
        return self.model

   

