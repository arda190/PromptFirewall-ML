import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path="datasets/raw/combined_dataset.csv"):
    data = pd.read_csv(path)

    text = data["text"].tolist()
    label = data["label"].tolist()


    X_train , X_test , y_train , y_test = train_test_split(
        text,
        label,
        test_size=0.2,
        stratify=label,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test