import pytest
import pandas as pd

from src.models.predict import predict_sentiment_topic
from src.preprocessing.transformations import apply_cleaning
from src.preprocessing.preprocessing_utils import remove_punctuations_df, remove_trailing_leading_spaces_df, replace_multiple_spaces_df, strip_html_tags_df

def test_cleaning_when_date_is_string():
    df = pd.DataFrame([["positive", "18/6/21", "I love the pizza here."]], columns = ["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame([["18/6/21", "I love the pizza here.", 1, "love pizza"]], columns = ['date', 'partially_cleaned_text', 'sentiment', 'cleaned_text'])
    return pd.testing.assert_frame_equal(apply_cleaning(df = df), df_expected_output, check_index_type=False)

def test_cleaning_when_date_is_datetime():
    df = pd.DataFrame([["positive", "18/6/21", "I love the pizza here."]], columns = ["Sentiment", "Time", "Text"])
    df["Time"] = pd.to_datetime(df["Time"])
    df_expected_output = pd.DataFrame([["18/6/21", "I love the pizza here.", 1, "love pizza"]], columns = ['date', 'partially_cleaned_text', 'sentiment', 'cleaned_text'])
    df_expected_output = df_expected_output.astype({'date': 'datetime64[ns]'})
    return pd.testing.assert_frame_equal(apply_cleaning(df = df), df_expected_output, check_index_type=False)

def test_cleaning_punctuation():
    df = pd.DataFrame([["positive", "18/6/21", "This is the best pizza ever!!!!@#$%^~*()'''''>>>>"]], columns = ["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive", "18/6/21", "This is the best pizza ever                      "]], columns = ["Sentiment", "Time", "cleaned_text"])
    return pd.testing.assert_frame_equal(remove_punctuations_df(df = df), df_expected_output, check_index_type=False)

def test_cleaning_remove_excess_spaces_for_tabs(): #checks if tabs are accounted for in trailing spaces
    df = pd.DataFrame([["positive", "18/6/21", "This is the best pizza ever wow     amazing"]], columns = ["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive", "18/6/21", "This is the best pizza ever wow amazing"]], columns = ["Sentiment", "Time", "cleaned_text"])
    return pd.testing.assert_frame_equal(replace_multiple_spaces_df(df = df), df_expected_output, check_index_type=False)

def test_cleaning_punctuation_and_html_tags(): #checks if punctuation and html tags removal interfere with eachother
    df = pd.DataFrame([["positive", "18/6/21", "<p>This is the best pizza ever </p><br><br>"]], columns = ["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive", "18/6/21", " This is the best pizza ever    "]], columns = ["Sentiment", "Time", "cleaned_text"])
    return pd.testing.assert_frame_equal(remove_punctuations_df(strip_html_tags_df(df = df)), df_expected_output, check_index_type=False)

def test_predict_when_null_reviews():
    df = pd.DataFrame(columns = ["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame(columns = ['date', 'partially_cleaned_text', 'sentiment', 'cleaned_text'])
    return pd.testing.assert_frame_equal(predict_sentiment_topic(test_filepath = "", df = df), df_expected_output, check_index_type=False)

def test_predict_when_all_stopwords():
    df = pd.DataFrame([["positive", "18/6/21", "I am"]], columns = ["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame(columns = ['date', 'partially_cleaned_text', 'sentiment', 'cleaned_text'])
    df_expected_output = df_expected_output.astype({'sentiment': 'int64'})
    return pd.testing.assert_frame_equal(predict_sentiment_topic(test_filepath = "", df = df), df_expected_output, check_index_type = False)

def test_predict_when_empty_review():
    df = pd.DataFrame([["positive", "18/6/21", ""]], columns = ["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame(columns = ['date', 'partially_cleaned_text', 'sentiment', 'cleaned_text'])
    df_expected_output = df_expected_output.astype({'sentiment': 'int64'})
    return pd.testing.assert_frame_equal(predict_sentiment_topic(test_filepath = "", df = df), df_expected_output, check_index_type = False)


# test cleaning html tags
# test app response if no dataframe
# test app response if non csv uploaded
# test app response if column names wrong
# 

if __name__ == "__main__":
    print(test_cleaning_when_date_is_string())
    print(test_cleaning_when_date_is_datetime())
    print(test_cleaning_punctuation())
    print(test_cleaning_punctuation_and_html_tags())
    print(test_cleaning_remove_excess_spaces_for_tabs())
    print(test_predict_when_null_reviews())
    print(test_predict_when_all_stopwords())
    print(test_predict_when_empty_review())

