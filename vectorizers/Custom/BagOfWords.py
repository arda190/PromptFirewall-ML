class BagOfWords:
    def __init__(self):
        self.words = []
        self.dictionary = {}


    def createDictionary(self,data): # data is a list []
        vocabulary = {}

        i = 0
        for sentence in data:
           for word in sentence.split():
               if word not in vocabulary:
                   vocabulary[word] = i
                   i+=1
        self.dictionary = vocabulary



    def fit(self,x_): #x_ is a list []
        self.createDictionary(x_)
        self.words = x_



    def transform(self,x_):
        features = []

        for sentence in x_:
            feature = [0] * len(self.dictionary)
            for word in sentence.split():
                if word in self.dictionary:
                    index = self.dictionary[word]
                    feature[index] += 1

            features.append(feature)

        return features


    def fit_transform(self,x_):
        self.fit(x_)

        return self.transform(x_)



































