from src.models.classifier import Classifier

class NonMatrixFactorization(Classifier):
    def fit(self, df, column):
        x = 2