import json
import math
class LogisticRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.bias = 0
        self.weights = []


    def sigmoid(self,x):
        return 1/(1+math.exp(-x))


    def saveToFile(self):
        pass


    def predict(self,x):
        prediction = self.bias

        for x1,w1 in zip(x,self.weights):
            prediction += w1*x1

        return self.sigmoid(prediction)


    def predict_class(self, x):
        prediction =self.predict(x)

        if prediction > 0.5:
            return 1
        else:
            return 0




    def fit(self,x_,y_):
        self.weights = [0] * len(x_[0])  # [ [1,1,3,4,2] , [6,7,8,9,4]  ]   [  [2,2,4,5,3] , [7,8,9,10,5] ]

        for epoch in range(1000):

           dw = [0] * len(x_[0])
           db = 0

           for x , actual_y in zip(x_,y_):
               # logistic regression use log loss

              p=self.predict(x)

              for i in range(len(x)):
                  dw[i] += (p - actual_y) * x[i]

              db += (p - actual_y)

           db /=len(x_)
           self.bias = self.bias - self.learning_rate * db

           for i in range(len(self.weights)):
               dw[i] /= len(x_)
               self.weights[i] =self.weights[i] - self.learning_rate * dw[i]

    def save_model(self, filename):
        data = {
            "weights": self.weights,
            "bias": self.bias
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_model(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
            self.weights = data["weights"]
            self.bias = data["bias"]
