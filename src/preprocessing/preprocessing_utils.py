import pandas as pd
import re

def convert_sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["Sentiment"].apply(
        lambda x: 1 if x == "positive" else 0
        )
    return df

def strip_html_tags_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda text: re.sub('<[^<]+?>', ' ', text)
        )
    return df