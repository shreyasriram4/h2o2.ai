from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def fit(self, *params):
        pass

    @abstractmethod
    def predict(self, *params):
        pass

    @abstractmethod
    def evaluate(self, *params):
        pass
