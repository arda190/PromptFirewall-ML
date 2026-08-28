import os
from data_loader import load_dataset
from vectorizers.library.factory import get_vectorizer




def train_model(vectorizer_name, model,dataset ="datasets/raw/combined_dataset.csv" ):
    X_train , _ , y_train , _ = load_dataset(dataset)

    vectorizer = get_vectorizer(vectorizer_name)

    X_vector = vectorizer.transform(X_train)

    model.fit(X_vector, y_train)

    name = vectorizer_name + "_" + model.name

    path = os.path.join("saved_models", name)

    model.save_model(path)
