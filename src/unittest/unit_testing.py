import pandas as pd

from src.utils.file_util import FileUtil
from src.models.predict import predict_sentiment_topic
from src.models.topic_modelling.test.predict import predict_topic
from src.models.sentiment_analysis.test.predict import predict_sentiment
from src.preprocessing.transformations import (apply_cleaning_test,
                                               apply_cleaning_train)
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

files = FileUtil()
config_params = files.CONFIG_PARAMS
topics = config_params["topics"]
subtopics = sum(config_params["topic_mapping"].values(), [])

# Testing preprocessing functions


def test_convert_sentiment_df():
    df = pd.DataFrame([["positive", "18/6/21", "I love the pizza here."]],
                      columns=["Sentiment", "Time", "Text"])
    df_expected_output = pd.DataFrame([["18/6/21",
                                        "I love the pizza here.",
                                        1]],
                                      columns=["Time", "Text", "sentiment"])
    return pd.testing.assert_frame_equal(convert_sentiment_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_expand_contractions_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "I'm They'd can't shouldn't should've"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
        [["positive",
          "18/6/21",
          "I am They would cannot should not should have"]],
        columns=["Sentiment", "Time", "cleaned_text"])
    return pd.testing.assert_frame_equal(expand_contractions_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_lowercase_string_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "THIS IS REally exciting. fOODd"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
        [["positive",
          "18/6/21",
          "this is really exciting. foodd"]],
        columns=["Sentiment", "Time", "cleaned_text"])
    return pd.testing.assert_frame_equal(lowercase_string_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_remove_numbers_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "h3110! 1 am 3ating food 123a4.3"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
        [["positive",
          "18/6/21",
          "h!  am ating food a."]],
        columns=["Sentiment", "Time", "cleaned_text"])
    return pd.testing.assert_frame_equal(remove_numbers_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_remove_punctuations_df():
    df = pd.DataFrame([["positive", "18/6/21",
                        "Thi}s is the b,,est pi!zza ever!@#$%^~*()'>"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
        [["positive",
          "18/6/21",
          "Thi s is the b  est pi zza ever            "]],
        columns=["Sentiment",
                 "Time",
                 "cleaned_text"])
    return pd.testing.assert_frame_equal(remove_punctuations_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_remove_stopwords_df():
    df = pd.DataFrame([["positive", "18/6/21",
                        "I am can try to eat a the out other by during"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
        [["positive",
          "18/6/21",
          "I try eat"]],
        columns=["Sentiment",
                 "Time",
                 "cleaned_text"])
    return pd.testing.assert_frame_equal(remove_stopwords_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_remove_trailing_leading_spaces_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "  This is the best pizza ever     wow "]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame(
        [["positive",
          "18/6/21",
          "This is the best pizza ever     wow"]],
        columns=["Sentiment",
                 "Time",
                 "cleaned_text"])
    return pd.testing.assert_frame_equal(
                            remove_trailing_leading_spaces_df(df=df),
                            df_expected_output,
                            check_index_type=False)


def test_rename_column_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "This is the best pizza ever"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive",
                                        "18/6/21",
                                        "This is the best pizza ever"]],
                                      columns=["sentiment",
                                               "Time",
                                               "cleaned_text"])
    return pd.testing.assert_frame_equal(rename_column_df(df=df,
                                                          src_col="Sentiment",
                                                          dst_col="sentiment"),
                                         df_expected_output,
                                         check_index_type=False)


def test_replace_multiple_spaces_df():
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

# checks if tabs are accounted for in trailing spaces


def test_replace_multiple_spaces_df_for_tabs():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "This is the best  .  pizza ever  wow"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive",
                                        "18/6/21",
                                        "This is the best . pizza ever wow"]],
                                      columns=["Sentiment",
                                               "Time",
                                               "cleaned_text"])
    return pd.testing.assert_frame_equal(replace_multiple_spaces_df(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_strip_html_tags_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        "<br>Hi there<h1>Title</h1>"]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([["positive",
                                        "18/6/21",
                                        " Hi there Title "]],
                                      columns=["Sentiment",
                                               "Time",
                                               "cleaned_text"])
    return pd.testing.assert_frame_equal(strip_html_tags_df(df=df),
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


def test_remove_empty_reviews_df():
    df = pd.DataFrame([["positive",
                        "18/6/21",
                        ""]],
                      columns=["Sentiment", "Time", "cleaned_text"])
    df_expected_output = pd.DataFrame([],
                                      columns=["Sentiment",
                                               "Time",
                                               "cleaned_text"])
    return pd.testing.assert_frame_equal(
        remove_empty_reviews_df(df=df),
        df_expected_output, check_index_type=False)


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


def test_apply_cleaning_test():
    df = pd.DataFrame([["18/6/21", " I love the pizza here."],
                       ["12/6/20", "<h2> Try this   asap.</h2>"],
                       ["12/12/13", "I HATE THIS FOOD12345"]],
                      columns=["Time", "Text"])
    df["Time"] = pd.to_datetime(df["Time"])
    df_expected_output = pd.DataFrame([["18/6/21",
                                        "I love the pizza here.",
                                        "love pizza"],
                                       ["12/6/20",
                                        "Try this asap.",
                                        "try soon possible"],
                                       ["12/12/13",
                                        "I HATE THIS FOOD12345",
                                        "hate food"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    df_expected_output = df_expected_output.astype({'date': 'datetime64[ns]'})
    return pd.testing.assert_frame_equal(apply_cleaning_test(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_apply_cleaning_train():
    df = pd.DataFrame([["positive", "18/6/21", " I love the pizza here."],
                       ["negative", "12/6/20", "<h2> Try this   asap.</h2>"],
                       ["positive", "12/12/13", "I HATE THIS FOOD12345"]],
                      columns=["Sentiment", "Time", "Text"])
    df["Time"] = pd.to_datetime(df["Time"])
    df_expected_output = pd.DataFrame([["18/6/21",
                                        "I love the pizza here.",
                                        1,
                                        "love pizza"],
                                       ["12/6/20",
                                        "Try this asap.",
                                        0,
                                        "try soon possible"],
                                       ["12/12/13",
                                        "I HATE THIS FOOD12345",
                                        1,
                                        "hate food"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'sentiment',
                                               'cleaned_text'])
    df_expected_output = df_expected_output.astype({'date': 'datetime64[ns]'})
    return pd.testing.assert_frame_equal(apply_cleaning_train(df=df),
                                         df_expected_output,
                                         check_index_type=False)


def test_predict_when_null_reviews():
    df = pd.DataFrame(columns=["Time", "Text"])
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
    df = pd.DataFrame([["18/6/21", "I am"]],
                      columns=["Time", "Text"])
    df_expected_output = pd.DataFrame(columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    return pd.testing.assert_frame_equal(
        predict_sentiment_topic(
            test_filepath="",
            df=df),
        df_expected_output,
        check_index_type=False)


def test_predict_when_empty_review():
    df = pd.DataFrame([["18/6/21", ""]],
                      columns=["Time", "Text"])
    df_expected_output = pd.DataFrame(columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    return pd.testing.assert_frame_equal(
        predict_sentiment_topic(
            test_filepath="",
            df=df),
        df_expected_output,
        check_index_type=False)

# testing prediction functions

# check that sentiments are either 0 or 1


def test_predict_sentiment():
    df = pd.DataFrame([["18/6/21", "I am so happy!"],
                       ["19/2/21", "this was disappointing."]],
                      columns=["Time", "Text"])
    df_expected_output = pd.DataFrame([["18/6/21", "I am so happy!",
                                        "happy"],
                                       ["19/2/21",
                                        "this was disappointing.",
                                        "disappointing"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    output = predict_sentiment(df=apply_cleaning_test(df))

    if not all(sentiment in [0, 1] for sentiment in list(output["sentiment"])):
        check = "Sentiment output values are not binary (strictly 0 or 1)"
        return check
    else:
        return pd.testing.assert_frame_equal(output[["date",
                                                     "partially_cleaned_text",
                                                     "cleaned_text"]],
                                             df_expected_output,
                                             check_index_type=False)

# check that topics and subtopics are within candidate labels


def test_predict_topic():
    check = None
    df = pd.DataFrame([["18/6/21", "these chips were bad."],
                       ["19/2/21", "Such good coffee!"]],
                      columns=["Time", "Text"])
    output = predict_topic(df=apply_cleaning_test(df))

    if not all(topic in topics for topic in list(output["topic"])):
        check = "Output topics are not in list of possible topic labels"
    if not all(subtopic in subtopics for subtopic in list(output["subtopic"])):
        check = "Output subtopics are not in list of possible subtopic labels"

    df_expected_output = pd.DataFrame([["18/6/21", "these chips were bad.",
                                        "chips bad"],
                                       ["19/2/21",
                                        "Such good coffee!",
                                        "good coffee"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    if check:
        return check
    else:
        return pd.testing.assert_frame_equal(
            output[['date',
                    'partially_cleaned_text',
                    'cleaned_text']],
            df_expected_output,
            check_index_type=False)

# check that sentiments are either 0 or 1
# check that sentiment probabilities are between 0 and 1
# check that topics and subtopics are within candidate labels


def test_predict_sentiment_topic():
    check = None
    df = pd.DataFrame([["18/6/21", "these chips are bad."],
                       ["19/2/21", "Such good coffee!"]],
                      columns=["Time", "Text"])

    output = predict_sentiment_topic(df=df)

    if not output["sentiment_prob"].between(0, 1).all():
        check = "Sentiment probabilities are out of range of 0 to 1"
    if not all(sentiment in [0, 1] for sentiment in list(output["sentiment"])):
        check = "Sentiment output values are not binary (strictly 0 or 1)"
    if not all(topic in topics for topic in list(output["topic"])):
        check = "Output topics are not in list of possible topic labels"
    if not all(subtopic in subtopics for subtopic in list(output["subtopic"])):
        check = "Output subtopics are not in list of possible subtopic labels"

    df_expected_output = pd.DataFrame([["18/6/21", "these chips are bad.",
                                        "chips bad"],
                                       ["19/2/21",
                                        "Such good coffee!",
                                        "good coffee"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    if check:
        return check
    else:
        return pd.testing.assert_frame_equal(
            output[['date',
                    'partially_cleaned_text',
                    'cleaned_text']],
            df_expected_output,
            check_index_type=False)

# test lbl2vec module


def test_lbl2vec_module():
    pass


def test_zeroshot_module():
    pass


def test_lda_module():
    pass


def test_nmf_module():
    pass


def test_bertopic_module():
    pass


def test_topic_modelling_train_module():
    pass


def test_bert_module():
    pass


def test_lstm_module():
    pass


def test_logreg_module():
    pass


def test_sentiment_analysis_train_module():
    pass

# TEST FILE UTIL ????

def unit_test():
    # Testing preprocessing utils
    test_convert_sentiment_df()
    test_expand_contractions_df()
    test_lowercase_string_df()
    test_remove_numbers_df()
    test_remove_punctuations_df()
    test_remove_stopwords_df()
    test_remove_trailing_leading_spaces_df()
    test_rename_column_df()
    test_replace_multiple_spaces_df()
    test_strip_html_tags_df()
    test_remove_empty_reviews_df()

    # Testing cleaning functions
    test_apply_cleaning_test()
    test_apply_cleaning_train()

    # Testing cleaning edge cases
    test_cleaning_when_date_is_string()
    test_cleaning_when_date_is_datetime()
    test_cleaning_punctuation_and_html_tags()
    test_replace_multiple_spaces_df_for_tabs()

    # Testing prediction functions

    test_predict_sentiment()
    test_predict_sentiment_topic()
    test_predict_topic()

    # Testing predict function edge cases
    test_predict_when_null_reviews()
    test_predict_when_all_stopwords()
    test_predict_when_empty_review()

    # Testing model-specific prediction functions

    # Test model-specific training functions

    # Test FileUtil module


if __name__ == "__main__":
    unit_test()
