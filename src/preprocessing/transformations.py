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


def main():
    df = FileUtil.get_raw_train_data()
    df = apply_cleaning_train(df)

    FileUtil.put_processed_train_data(df)


def apply_cleaning_train(df: pd.DataFrame) -> pd.DataFrame:
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
    main()
