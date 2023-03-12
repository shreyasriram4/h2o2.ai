import os
import pandas as pd

from src.preprocessing.preprocessing_utils import (
    convert_sentiment_df,
    expand_contractions_df,
    lowercase_string_df,
    remove_numbers_df,
    remove_punctuations_df,
    remove_stopwords_df,
    remove_trailing_leading_spaces_df,
    rename_column_df,
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

    create_path_if_not_exists(PROCESSED_DATA_DIR)
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, FILE_NAME))

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        .pipe(remove_trailing_leading_spaces_df)
        .pipe(rename_column_df, "Time", "date")
        .pipe(rename_column_df, "Text", "partially_cleaned_text")
    )

if __name__ == "__main__":
    main()
