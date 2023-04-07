import pandas as pd

from src.utils.file_util import FileUtil
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
    strip_html_tags_df,
    remove_empty_reviews_df
)


def preprocess_train():
    """
    Applies cleaning to raw training data in according
    to filepath specified in FileUtil module.
    Returns processed training data to filepath specified
    in FileUtil module.
    """
    df = FileUtil.get_raw_train_data()
    df = apply_cleaning_train(df)

    FileUtil.put_processed_train_data(df)


def apply_cleaning_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning to training data as follows:
    1. Converts sentiment column to binary integer format
    2. Removes HTML tags, multiple spaces and trailing leading spaces
    to form partially_cleaned_text column.
    3. Removes HTML tags, multiple spaces, trailing leading spaces,
    stopwords and numbers. Lowercases text, expands contractions and
    removes empty reviews to form cleaned_text column.
    4. Renames Time column to date

    Args:
        df (pd.Dataframe): input dataframe with columns Text,
        Sentiment and Time

    Returns:
        df (pd.Dataframe): dataframe consisting of new text columns
        partially_cleaned_text and cleaned_text.
    """
    return (
        df.pipe(convert_sentiment_df)
        .pipe(strip_html_tags_df, src_col="Text", dst_col="Text")
        .pipe(replace_multiple_spaces_df, src_col="Text", dst_col="Text")
        .pipe(remove_trailing_leading_spaces_df,
              src_col="Text",
              dst_col="Text")
        .pipe(lowercase_string_df, src_col="Text")
        .pipe(expand_contractions_df)
        .pipe(remove_numbers_df)
        .pipe(remove_punctuations_df)
        .pipe(remove_stopwords_df)
        .pipe(replace_multiple_spaces_df)
        .pipe(remove_trailing_leading_spaces_df)
        .pipe(remove_empty_reviews_df)
        .pipe(rename_column_df, "Time", "date")
        .pipe(rename_column_df, "Text", "partially_cleaned_text")
    )


def apply_cleaning_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning to test data as follows:
    1. Removes HTML tags, multiple spaces and trailing leading spaces
    to form partially_cleaned_text column.
    2. Removes HTML tags, multiple spaces, trailing leading spaces,
    stopwords and numbers. Lowercases text, expands contractions and
    removes empty reviews to form cleaned_text column.
    3. Renames Time column to date

    Args:
        df (pd.Dataframe): input dataframe with columns Text and Time

    Returns:
        df (pd.Dataframe): dataframe consisting of new text columns
        partially_cleaned_text and cleaned_text.
    """
    return (
        df.pipe(strip_html_tags_df, src_col="Text", dst_col="Text")
        .pipe(replace_multiple_spaces_df, src_col="Text", dst_col="Text")
        .pipe(remove_trailing_leading_spaces_df,
              src_col="Text",
              dst_col="Text")
        .pipe(lowercase_string_df, src_col="Text")
        .pipe(expand_contractions_df)
        .pipe(remove_numbers_df)
        .pipe(remove_punctuations_df)
        .pipe(remove_stopwords_df)
        .pipe(replace_multiple_spaces_df)
        .pipe(remove_trailing_leading_spaces_df)
        .pipe(remove_empty_reviews_df)
        .pipe(rename_column_df, "Time", "date")
        .pipe(rename_column_df, "Text", "partially_cleaned_text")
    )


if __name__ == "__main__":
    preprocess_train()
