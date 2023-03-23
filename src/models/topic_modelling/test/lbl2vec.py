from lbl2vec import Lbl2TransformerVec
from src.models.classifier import Classifier

class Lbl2Vec(Classifier):
    def fit(self, df, column, candidate_labels):
        model = Lbl2TransformerVec(
            keywords_list = list(candidate_labels.values()), 
            documents = list(df[column])
            )
        
        model.fit()

        return model

    def predict(self, df, column, candidate_labels):
        preds = self.fit(df, column, candidate_labels).predict_model_docs()
        df["topic"] = preds['most_similar_label']

        df["topic"] = df["topic"].apply(lambda x: list(candidate_labels.keys())[int(x[-1])])

        return df

    def evaluate(self):
        pass