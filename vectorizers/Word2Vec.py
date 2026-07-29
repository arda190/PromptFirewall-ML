from gensim.models import Word2Vec
import numpy as np

class Word2Vectorization:
    def __init__(self):
        self.model = None



    def fit(self,sentences):
        sentences = [sentence.split() for sentence in sentences]
        self.model = Word2Vec(
            sentences,
            vector_size=100,
            window=5,
            min_count=1
        )

    def __sentence_vector(self,sentence):
        vectors = []

        for word in sentence:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])

        return np.mean(vectors, axis=0)


    def transform(self,sentences):
        vectors = []
        for sentence in sentences:
            vectors.append(self.__sentence_vector(sentence))

        return vectors


    def fit_transform(self,sentences):
        self.fit(sentences)
        return self.transform(sentences)



    def save_model(self,filename):
        self.model.save(filename)


    def load_model(self,path):
        self.model=Word2Vec.load(path)