from src.models.sentiment_analysis.train.bert import BERT


def predict_sentiment(df):
    model = BERT(True)

    label, probs, tf_predictions = model.predict(df)

    df["sentiment"] = label
    df["sentiment_prob"] = probs

    return df
