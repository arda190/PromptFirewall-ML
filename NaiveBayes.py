import math

class NaiveBayes:
    def __init__(self):
        self.spam_prob = []
        self.ham_prob = []
        self.positive = 0  # harmful prompt
        self.negative = 0  # harmless prompt
        self.spam_count = []
        self.ham_count = []
        self.spam_doc = 0
        self.ham_doc = 0
        self.spam_voc_size = 0
        self.ham_voc_size = 0


    def classify(self,Y):
        for i in range(len(Y)):
            if Y[i] == 1:
                self.spam_doc+=1
            else:
                self.ham_doc+=1

    def calculateProbabilitySpam(self,x):
        spam_probability = self.spam_doc / ( self.spam_doc + self.ham_doc )

        for index,i in enumerate(x):
            if i!=0:
                spam_probability *= math.pow(self.spam_prob[index],i)

        return spam_probability


    def calculateProbabilityHam(self,x):
        ham_probability = self.ham_doc / (self.spam_doc + self.ham_doc)

        for index, i in enumerate(x):
            if i != 0:
                ham_probability *= math.pow(self.ham_prob[index],i)

        return ham_probability


    def predict(self,x):

        spam_prob = self.calculateProbabilitySpam(x)
        ham_prob = self.calculateProbabilityHam(x)

        if spam_prob >=ham_prob:
            return 1
        elif spam_prob < ham_prob:
            return 0

        return None

    def fit(self, X, Y):

        length_X = len(X[0])

        self.classify(Y)

        self.spam_count = [0] * length_X
        self.ham_count = [0] * length_X


        for x,y in zip(X,Y):

           for i,count in enumerate(x):
               if y == 0:
                   self.ham_count[i]+=count
                   self.negative+=count
                   self.ham_voc_size += 1
               elif y == 1:
                   self.spam_count[i]+=count
                   self.positive+=count
                   self.spam_voc_size += 1



        self.spam_prob = [0] * length_X
        self.ham_prob = [0] * length_X



        for i in range(len(X[0])):
            self.spam_prob[i] = ( self.spam_count[i] + 1 ) / ( self.positive + length_X )
            self.ham_prob[i] = ( self.ham_count[i] + 1 ) / ( self.negative + length_X )








