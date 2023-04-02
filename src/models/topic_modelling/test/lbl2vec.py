from lbl2vec import Lbl2TransformerVec

from src.models.classifier import Classifier


class Lbl2Vec(Classifier):
    def fit(self, df, column, candidate_labels):
        model = Lbl2TransformerVec(
            keywords_list=list(map(lambda subtopic: [subtopic],
                                   candidate_labels.keys())),
            documents=list(df[column])
            )

        model.fit()

        return model

    def predict(self, df, column, candidate_labels):
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
        pass
