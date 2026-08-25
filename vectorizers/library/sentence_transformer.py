from sentence_transformers import SentenceTransformer
import joblib

class SentenceTransformerWrapper:
    def __init__(self,model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)


    def transform(self,X,batch_size=16):
        return self.model.encode(X,batch_size=batch_size,show_progress_bar=True)







