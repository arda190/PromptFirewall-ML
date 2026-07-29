class LinearRegression:
    def __init__(self):
        self.bias=0
        self.weight=0
        self.learning_rate=0.001

    def predict(self, x):
        prediction=self.bias + self.weight * x
        return prediction


    def train(self,x_,y_):

        for epoch in range(8000):
          #print(epoch,self.weight,self.bias)
          dw = 0
          db = 0

          for x,actual_y in zip(x_,y_):
             prediction=self.predict(x)

             error=prediction-actual_y

             loss=error**2



             dw=dw+((error)*x*2)
             db=db+(error) * 2

          dw /= len(x_)
          db /= len(x_)

          self.weight = self.weight - self.learning_rate * dw
          self.bias = self.bias  - self.learning_rate * db

