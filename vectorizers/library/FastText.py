from gensim.models import FastText
import numpy as np

class FastTextVectorizer:
    def __init__(self,vector_size=100):
        self.model=None
        self.vector_size =vector_size
        self.name = "FastText"



    def fit(self,sentences):
        tokenized = [sentence.split() for sentence in sentences]
        self.model = FastText(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=5,
            min_count=1
        )


    def transform(self,sentences):

        if self.model is None:
            raise ValueError("FastTextVectorizer has not been trained yet.")

        vectors = []

        for sentence in sentences:
            word_vectors = []

            for word in sentence.split():
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])

            if len(word_vectors) ==0:
                vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(np.mean(word_vectors,axis=0))

        return vectors



    def fit_transform(self,sentences):
        self.fit(sentences)
        return self.transform(sentences)




    def save_model(self,filename):
        self.model.save(filename)


    def load_model(self,filename="saved_models/fasttext"):
        self.model=FastText.load(filename)
        print("model loaded")
        return self





