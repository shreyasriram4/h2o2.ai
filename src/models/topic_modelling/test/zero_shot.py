"""This module contains ZeroShot topic model class."""

import os
import pandas as pd
from transformers import pipeline

from src.models.classifier import Classifier
from src.utils.file_util import FileUtil


class ZeroShot(Classifier):
    """ZeroShot topic model class."""

    def save_model(self):
        """Save ZeroShot model to storage."""
        classifier = pipeline(task="zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=-1)
        FileUtil.put_topic_model(classifier)

    def get_model(self):
        """Get ZeroShot model from storage."""
        model_file_path = os.path.join(FileUtil().TOPIC_MODELLING_DIR,
                                       FileUtil().MODEL_FILE_NAME)
        if not FileUtil.check_filepath_exists(model_file_path):
            self.save_model()
        return FileUtil.get_topic_model()

    def dataloader(self, df, column):
        """
        Generate each row text from df.

        Args:
          df (pd.DataFrame): dataframe to retrieve text
          column (str): text column in df

        Yields:
          text value in column
        """
        for i in range(len(df)):
            yield df.loc[i, column]

    def predict(self, df, column, candidate_labels):
        """
        Predict the topic for df.

        Args:
          df (pd.DataFrame): dataframe to predict
          column (str): text column in df
          candidate_labels (list): list of topics

        Returns:
          df (pd.Dataframe): prediction result dataframe
        """
        clf = self.get_model()
        hypothesis_template = "The topic of this review is {}."

        preds = clf(
            self.dataloader(df, column),
            candidate_labels,
            hypothesis_template=hypothesis_template
            )

        preds = pd.DataFrame(preds)
        df['topic'] = preds['labels'].apply(lambda x: x[0])

        return df

    def fit(self):
        """Fit ZeroShot topic model."""
        pass

    def evaluate(self):
        """Evaluate topic classification."""
        pass


if __name__ == "__main__":
    ZeroShot().save_model()
