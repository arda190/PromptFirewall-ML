import math
from models.LogisticRegression import LogisticRegression




class TFIDF:
    def __init__(self):
        self.idfValues = {}
        self.vocabulary = {}


    def calculate_IDF(self,data,doc_count):

        idf = {}

        for word in data.keys() :
            idf[word] = math.log(doc_count/data[word])

        return idf




    def createIDF(self,data):

        idf_values = {}

        for sentence in data :
            unique_words = set(sentence.split())

            for word in unique_words :
                idf_values[word] = idf_values.get(word,0)+1


        self.idfValues = self.calculate_IDF(idf_values,len(data))



    def createVocabulary(self,data):

        vocabulary = {}

        for sentence in data :
            for word in sentence.split() :
                if word not in vocabulary :
                   vocabulary[word] = len(vocabulary)

        self.vocabulary = vocabulary



    def fit(self,data):

        self.createVocabulary(data)
        self.createIDF(data)



    def transform(self,data):

       transformed_data = []

       for sentence in data:
           word_count = {}
           total_words = len(sentence.split())
           feature = [0] * len(self.vocabulary)
           for word in sentence.split() :
               word_count[word] = word_count.get(word,0)+1


           for w in word_count.keys() :
               if w in self.vocabulary :
                  tf = word_count[w]/total_words
                  idf = self.idfValues[w]
                  p = tf * idf

                  index =self.vocabulary[w]

                  feature[index] = p
           transformed_data.append(feature)


       return transformed_data



    def fit_transform(self,data):
        self.fit(data)

        return self.transform(data)



