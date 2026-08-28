from train import train_model
from models.library.NaiveBayes import  NaiveBayesWrapper
from models.library.LogisticRegression import LogisticRegressionWrapper
from models.library.LightGBM import  LightGBMWrapper
from models.library.SupportVectorMachine import  SupportVectorMachineWrapper


models = [
    LogisticRegressionWrapper(),
    LightGBMWrapper(),
    SupportVectorMachineWrapper(),
    #NaiveBayesWrapper(),
]

for model in models:
    train_model("fasttext",model)