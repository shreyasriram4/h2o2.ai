import pandas as pd

def convert_sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["Sentiment"].apply(
        lambda x: 1 if x == "positive" else 0
        )
    return df