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



x_ = [1,2,3,4,5]
y_ = [3,5,7,9,11]

model=LinearRegression()

model.train(x_,y_)
print(model.weight)
print(model.bias)
print(model.predict(6))



x_2=[1,2,3,4,5,6,7]     #3x-1
y_2=[2,5,8,11,14,17,20]

model_new=LinearRegression()
model_new.train(x_2,y_2)
print("--------")

print(model_new.weight)
print(model_new.bias)
print(model_new.predict(11)) #32











