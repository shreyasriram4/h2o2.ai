import os
import pandas as pd

from src.preprocessing.preprocessing_utils import (
    convert_sentiment_df,
    expand_contractions_df,
    lowercase_string_df,
    remove_numbers_df,
    remove_punctuations_df,
    remove_stopwords_df,
    replace_multiple_spaces_df,
    strip_html_tags_df
)

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed")
FILE_NAME = "reviews.csv"

def main():
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, FILE_NAME))
    df = apply_cleaning(df)
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, FILE_NAME))

def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(convert_sentiment_df)
        .pipe(strip_html_tags_df, src_col = "Text", dst_col = "Text")
        .pipe(replace_multiple_spaces_df, src_col = "Text", dst_col = "Text")
        .pipe(lowercase_string_df, src_col = "Text")
        .pipe(expand_contractions_df)
        .pipe(remove_numbers_df)
        .pipe(remove_punctuations_df)
        .pipe(remove_stopwords_df)
        .pipe(replace_multiple_spaces_df)
    )

if __name__ == "__main__":
    main()