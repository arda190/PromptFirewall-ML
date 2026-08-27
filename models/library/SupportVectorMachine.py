from sklearn.svm import SVC
import joblib

class SupportVectorMachineWrapper:
    def __init__(self,kernel="linear"):
        self.kernel = kernel
        self.model = None


    def fit(self,x,y):
        self.model = SVC(kernel=self.kernel,probability=True)
        self.model.fit(x,y)


    def predict(self,x):
        return self.model.predict(x)


    def save_model(self,path):
        joblib.dump(self.model,path)


    def load_model(self,path):
        self.model = joblib.load(path)


    def score(self,x,y):
        return self.model.score(x,y)