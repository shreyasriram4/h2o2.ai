import pandas as pd

from src.models.predict import predict_sentiment_topic
from src.preprocessing.transformations import apply_cleaning_test, apply_cleaning_train
from src.preprocessing.preprocessing_utils import (
                                            remove_punctuations_df,
                                            remove_trailing_leading_spaces_df,
                                            replace_multiple_spaces_df,
                                            strip_html_tags_df
                                            )


def test_cleaning_when_date_is_string():
    df = pd.DataFrame([["positive", "18/6/21", "I love the pizza here."]],
                      columns=["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame([["18/6/21",
                                        "I love the pizza here.",
                                        1,
                                        "love pizza"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'sentiment',
                                               'cleaned_text'])
    return pd.testing.assert_frame_equal(apply_cleaning_train(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_cleaning_when_date_is_datetime():
    df = pd.DataFrame([["positive", "18/6/21", "I love the pizza here."]],
                      columns=["Sentiment", "Time", "Text"])
    df["Time"] = pd.to_datetime(df["Time"])
    df_expected_output = pd.DataFrame([["18/6/21",
                                        "I love the pizza here.",
                                        1,
                                        "love pizza"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'sentiment',
                                               'cleaned_text'])
    df_expected_output = df_expected_output.astype({'date': 'datetime64[ns]'})
    return pd.testing.assert_frame_equal(apply_cleaning_train(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_cleaning_punctuation():
    df = pd.DataFrame([["positive", "18/6/21",
                        "This is the best pizza ever!@#$%^~*()'>"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
                            [["positive",
                              "18/6/21",
                              "This is the best pizza ever            "]],
                            columns=["Sentiment",
                                     "Time",
                                     "cleaned_text"])
    return pd.testing.assert_frame_equal(remove_punctuations_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


# checks if tabs are accounted for in trailing spaces
def test_cleaning_remove_excess_spaces_for_tabs():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "This is the best pizza ever     wow"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive",
                                        "18/6/21",
                                        "This is the best pizza ever wow"]],
                                      columns=["Sentiment",
                                               "Time",
                                               "cleaned_text"])
    return pd.testing.assert_frame_equal(replace_multiple_spaces_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


# checks if punctuation and html tags removal interfere with eachother
def test_cleaning_punctuation_and_html_tags():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "<p>This is the best pizza ever </p><br><br>"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive",
                                        "18/6/21",
                                        " This is the best pizza ever    "]],
                                      columns=["Sentiment",
                                               "Time",
                                               "cleaned_text"])
    return pd.testing.assert_frame_equal(
                    remove_punctuations_df(strip_html_tags_df(df=df)),
                    df_expected_output, check_index_type=False)


def test_predict_when_null_reviews():
    df = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame(columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    return pd.testing.assert_frame_equal(
                                        predict_sentiment_topic(
                                                test_filepath="",
                                                df=df),
                                        df_expected_output,
                                        check_index_type=False)


def test_predict_when_all_stopwords():
    df = pd.DataFrame([["positive", "18/6/21", "I am"]],
                      columns=["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame(columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    #df_expected_output = df_expected_output.astype({'sentiment': 'int64'})
    return pd.testing.assert_frame_equal(
                                        predict_sentiment_topic(
                                                test_filepath="",
                                                df=df),
                                        df_expected_output,
                                        check_index_type=False)


def test_predict_when_empty_review():
    df = pd.DataFrame([["positive", "18/6/21", ""]],
                      columns=["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame(columns=['date',
                                               'partially_cleaned_text',
                                               'sentiment',
                                               'cleaned_text'])
    df_expected_output = df_expected_output.astype({'sentiment': 'int64'})
    return pd.testing.assert_frame_equal(
                                        predict_sentiment_topic(
                                                test_filepath="",
                                                df=df),
                                        df_expected_output,
                                        check_index_type=False)


# test cleaning html tags

if __name__ == "__main__":
    test_cleaning_when_date_is_string()
    test_cleaning_when_date_is_datetime()
    test_cleaning_punctuation()
    test_cleaning_punctuation_and_html_tags()
    test_cleaning_remove_excess_spaces_for_tabs()
    test_predict_when_null_reviews()
    test_predict_when_all_stopwords()
    test_predict_when_empty_review()
