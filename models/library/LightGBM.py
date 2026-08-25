import lightgbm as lgb
import joblib

class LightGBM:
    def __init__(self,n_estimators=300,num_leaves=31,learning_rate=0.05,random_state=42):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None



    def fit(self,X,y):
        self.model = lgb.LGBMClassifier(n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate,
                                        num_leaves=self.num_leaves,
                                        random_state=self.random_state,)

        self.model.fit(X,y)


    def predict(self,X):
        return self.model.predict(X)
    

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)