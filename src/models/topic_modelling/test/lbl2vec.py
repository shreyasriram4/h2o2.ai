"""This module contains Lbl2Vec topic model class"""

from lbl2vec import Lbl2TransformerVec

from src.models.classifier import Classifier


class Lbl2Vec(Classifier):
    """Lbl2Vec topic model class."""

    def fit(self, df, column, candidate_labels):
        """
        Fit Lbl2TransformerVec on df with candidate_labels.

        Args:
          df (pd.DataFrame): dataframe to fit
          column (str): text column in df
          candidate_labels (dict): dictionary of subtopic, topic
          mapping

        Returns:
          model: fitted Lbl2TransformerVec model
        """
        model = Lbl2TransformerVec(
            keywords_list=list(map(lambda subtopic: [subtopic],
                                   candidate_labels.keys())),
            documents=list(df[column])
            )

        model.fit()

        return model

    def predict(self, df, column, candidate_labels):
        """
        Predict the topic for df.

        Args:
          df (pd.DataFrame): dataframe to predict
          column (str): text column in df
          candidate_labels (list): list of lists
          containing subtopics

        Returns:
          dataframe (pd.Dataframe): prediction result dataframe
        """
        preds = self.fit(df, column, candidate_labels).predict_model_docs()
        df["subtopic"] = preds['most_similar_label']

        df["topic"] = df["subtopic"].apply(
            lambda x: list(candidate_labels.values())[int(x.split("_")[1])]
            )
        df["subtopic"] = df["subtopic"].apply(
            lambda x: list(candidate_labels.keys())[int(x.split("_")[1])]
            )

        return df

    def evaluate(self):
        """Evaluate topic classification."""
        pass
