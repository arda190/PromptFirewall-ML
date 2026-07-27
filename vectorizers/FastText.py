from gensim.models import FastText

class FastTextVectorizer:
    def __init__(self, vector_size=100):
        self.model = None
        self.vector_size = vector_size



    def fit(self, sentences):
        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=5,
            min_count=1
        )


    def transform(self, sentences):
        pass




    def save_model(self, filename):
        self.model.save(filename)


    def load_model(self, filename):
        self.model = FastText.load(filename)





