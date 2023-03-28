from src.models.sentiment_analysis.train.bert import BERT
from src.utils.file_util import FileUtil

model_name = FileUtil.get_config()["best_sentiment_analysis_model"]


def predict_sentiment(df, model_name=model_name):
    assert model_name in ["BERT", "Logistic Regression", "LSTM"]

    if model_name == "BERT":
        model = BERT(True)
        label, probs, tf_predictions = model.predict(df)
        df["sentiment"] = label
        df["sentiment_prob"] = probs
    elif model_name == "Logistic Regression":
        pass
    else:
        pass

    return df
