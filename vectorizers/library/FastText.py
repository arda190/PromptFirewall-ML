from gensim.models import FastText
import numpy as np

class FastTextVectorizer:

  def __init__(self, vector_size=100, window=5, min_count=1, workers=4, **kwargs):
    self.model = None
    self.vector_size = vector_size
    self.window = window
    self.min_count = min_count
    self.workers = workers
    self.extra_kwargs = kwargs
    self.name = "FastText"

  def fit(self, sentences:list[str])->"FastTextVectorizer" :
    tokenized = [sentence.split() for sentence in sentences]
    self.model = FastText(
        sentences=tokenized,
        vector_size=self.vector_size,
        window=self.window,
        min_count=self.min_count,
        workers=self.workers,
        **self.extra_kwargs,
    )
    return self

  def transform(self, sentences:list[str])->np.ndarray:
    if self.model is None:
      raise ValueError("FastTextVectorizer has not been trained yet.")

    vectors = []
    for sentence in sentences:
        words = sentence.split()
        if not words:
            vectors.append(np.zeros(self.vector_size))
            continue

        word_vectors = []
        for word in words:
            try:
                word_vectors.append(self.model.wv[word])
            except KeyError:
                pass

        if len(word_vectors) == 0:
            vectors.append(np.zeros(self.vector_size))
        else:
            vectors.append(np.mean(word_vectors, axis=0))

    return np.array(vectors)

  def fit_transform(self, sentences:list[str])->np.ndarray:
    self.fit(sentences)
    return self.transform(sentences)

  def save_model(self, filename:str)->None:
    if self.model is None:
        raise ValueError("FastTextVectorizer has not been trained yet.")
    self.model.save(filename)

  def load_model(self, filename="saved_models/fasttext")->"FastTextVectorizer":
    self.model = FastText.load(filename)

    self.vector_size = self.model.vector_size
    return self
