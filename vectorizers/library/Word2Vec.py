from gensim.models import Word2Vec
import numpy as np

class Word2Vectorizer:
    def __init__(self,vector_size=100,window=5,min_count=1,workers=4,**kwargs):
        self.model = None
        self.name = "Word2Vec"
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.kwargs = kwargs



    def fit(self,sentences:list[str])->"Word2Vectorizer":
        sentences = [sentence.split() for sentence in sentences]
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            **self.kwargs
        )
        return self

    def __sentence_vector(self, sentence:list[str])->np.ndarray:
        vectors = []

        for word in sentence:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])

        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)

        return np.mean(vectors, axis=0)


    def transform(self, sentences:list[str])->np.ndarray:
        if self.model is None:
            raise ValueError("Word2Vectorizer has not been fitted yet")

        vectors = [self.__sentence_vector(sentence.split()) for sentence in sentences]

        return np.array(vectors)


    def fit_transform(self,sentences:list[str])->np.ndarray:
        self.fit(sentences)
        return self.transform(sentences)



    def save_model(self,filename:str)->None:
        if self.model is None:
            raise ValueError("Word2Vectorizer has not been fitted yet")
        self.model.save(filename)


    def load_model(self,path="saved_models/word2vec")->"Word2Vectorizer":
        self.model=Word2Vec.load(path)
        self.vector_size = self.model.vector_size
        return self