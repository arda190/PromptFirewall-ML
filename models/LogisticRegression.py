import math
class LogisticRegression:
    def __init__(self):
        self.learning_rate=0.01
        self.bias=0
        self.weight=0


    def sigmoid(self,x):
        return  1/(1+math.exp(-x))        # 1/(1+e^-x)



    def predict(self,x):
        prediction = self.bias + self.weight * x

        return self.sigmoid(prediction)

    def classify(self,x):
       prediction = self.predict(x)
       if prediction>=0.5:
           return 1
       else:
           return 0



    def log_loss(self,p,y):
        return - ( y * math.log(p) + (1-y) * math.log(1-p) )


    def fit(self,x_,y_):


        for epoch in range(10000):
            dw=0
            db=0

            for x,actual_y in zip(x_,y_):
                prediction = self.predict(x)

                loss=self.log_loss(prediction,actual_y)

                dw = dw + ( x * (prediction - actual_y) )
                db = db + (prediction - actual_y)

            dw = dw/len(x_)
            db = db/len(y_)

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db


x_ = [1,2,3,4,5,6,7,8,9,10]

y_ = [0,0,0,1,0,1,1,1,1,1]

model = LogisticRegression()

model.fit(x_,y_)

x=model.predict(-5)
print(x)

#print(model.predict(10))
#print(model.classify(10))
print(model.classify(-5))
