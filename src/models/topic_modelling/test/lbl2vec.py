from lbl2vec import Lbl2TransformerVec

class Lbl2Vec():
    def predict(self, df, column, candidate_labels):
        model = Lbl2TransformerVec(
            keywords_list = list(candidate_labels.values()), 
            documents = list(df[column])
            )
        
        model.fit()

        preds = model.predict_model_docs()
        df["topic"] = preds['most_similar_label']

        df["topic"] = df["topic"].apply(lambda x: list(candidate_labels.keys())[int(x[-1])])

        return df
