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
            min_count=1,
            workers=4,
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
            words = sentence.split()
            self.__sentence_vector(words)

        if len(vectors)==0:
            return np.zeros(self.model.vector_size)

        return np.array(vectors)


    def fit_transform(self,sentences):
        self.fit(sentences)
        return self.transform(sentences)



    def save_model(self,filename):
        self.model.save(filename)


    def load_model(self,path):
        self.model=Word2Vec.load(path)