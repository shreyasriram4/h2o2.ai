"""This module contains Classifier abstract class."""

from abc import ABC, abstractmethod


class Classifier(ABC):
    """
    An abstract Classifier class.

    Provide abstraction for fit, predict, and evaluate methods.
    """

    @abstractmethod
    def fit(self, *params):
        """Fit the Classifier."""
        pass

    @abstractmethod
    def predict(self, *params):
        """Predict using Classifier."""
        pass

    @abstractmethod
    def evaluate(self, *params):
        """Evaluate prediction result."""
        pass
