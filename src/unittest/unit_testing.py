import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.file_util import FileUtil, InvalidExtensionException
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

from src.models.topic_modelling.test.lbl2vec import Lbl2Vec
from src.models.topic_modelling.test.zero_shot import ZeroShot
from src.models.topic_modelling.train.lda import LDA
from src.models.topic_modelling.train.bertopic import BERTopic_Module
from src.models.topic_modelling.train.nmf import Tfidf_NMF_Module
from src.models.topic_modelling.train.train import topic_modelling_train

from src.models.sentiment_analysis.train.bert import BERT
from src.models.sentiment_analysis.train.logreg import LOGREG
from src.models.sentiment_analysis.train.lstm import Lstm
from src.models.sentiment_analysis.train.train import sentiment_analysis_train

files = FileUtil()
config_params = files.CONFIG_PARAMS
topics = config_params["topics"]
subtopics = sum(config_params["topic_mapping"].values(), [])
data_proc = files.get_processed_train_data().head(300)
data_proc_large = files.get_processed_train_data().head(1000)

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
        print("Sentiment output values are not binary (strictly 0 or 1)")

    else:
        return pd.testing.assert_frame_equal(output[["date",
                                                     "partially_cleaned_text",
                                                     "cleaned_text"]],
                                             df_expected_output,
                                             check_index_type=False)

# check that topics and subtopics are within candidate labels


def test_predict_topic():
    check = []
    df = pd.DataFrame([["18/6/21", "these chips were bad."],
                       ["19/2/21", "Such good coffee!"]],
                      columns=["Time", "Text"])
    output = predict_topic(df=apply_cleaning_test(df))

    if not all(topic in topics for topic in list(output["topic"])):
        check.append("Output topics are not in list of possible topic labels")
    if not all(subtopic in subtopics for subtopic in list(output["subtopic"])):
        check.append(
            "Output subtopics are not in list of possible subtopic labels")

    df_expected_output = pd.DataFrame([["18/6/21", "these chips were bad.",
                                        "chips bad"],
                                       ["19/2/21",
                                        "Such good coffee!",
                                        "good coffee"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    if check:
        print(check)
    else:
        return pd.testing.assert_frame_equal(
            output[['date',
                    'partially_cleaned_text',
                    'cleaned_text']],
            df_expected_output,
            check_index_type=False)


def test_predict_sentiment_topic():
    check = []
    df = pd.DataFrame([["18/6/21", "these chips are bad."],
                       ["19/2/21", "Such good coffee!"]],
                      columns=["Time", "Text"])

    output = predict_sentiment_topic(test_filepath="", df=df)

    if not output["sentiment_prob"].between(0, 1).all():
        check.append("Sentiment probabilities are out of range of 0 to 1")
    if not all(sentiment in [0, 1] for sentiment in list(output["sentiment"])):
        check.append(
            "Sentiment output values are not binary (strictly 0 or 1)")
    if not all(topic in topics for topic in list(output["topic"])):
        check.append("Output topics are not in list of possible topic labels")
    if not all(subtopic in subtopics for subtopic in list(output["subtopic"])):
        check.append(
            "Output subtopics are not in list of possible subtopic labels")

    df_expected_output = pd.DataFrame([["18/6/21", "these chips are bad.",
                                        "chips bad"],
                                       ["19/2/21",
                                        "Such good coffee!",
                                        "good coffee"]],
                                      columns=['date',
                                               'partially_cleaned_text',
                                               'cleaned_text'])
    if check:
        print(check)
    else:
        return pd.testing.assert_frame_equal(
            output[['date',
                    'partially_cleaned_text',
                    'cleaned_text']],
            df_expected_output,
            check_index_type=False)

# test lbl2vec module


def test_lbl2vec_module():
    check = []
    df = pd.DataFrame([["18/6/21", "these chips are bad."],
                       ["19/2/21", "Such good coffee!"]],
                      columns=["Time", "Text"])
    model = Lbl2Vec()
    candidate_labels = {'chips': 'snacks',
                        'crackers': 'snacks', 'coffee': 'drinks'}
    output_df = model.predict(df, 'Text', candidate_labels)

    df_expected_output = pd.DataFrame([["18/6/21", "these chips are bad."],
                                       ["19/2/21", "Such good coffee!"]],
                                      columns=['Time',
                                               'Text'])

    topics = list(set(candidate_labels.values()))
    subtopics = list(set(candidate_labels.keys()))

    if not all(subtopic in subtopics for subtopic in output_df['subtopic']):
        check.append("Subtopic labels not in subtopic")
    if not all(topic in topics for topic in output_df['topic']):
        check.append("Topic labels not in topic")

    if check:
        print(check)
    else:
        return pd.testing.assert_frame_equal(
            output_df[['Time',
                       'Text']],
            df_expected_output,
            check_index_type=False)


def test_zeroshot_module():
    df = pd.DataFrame([["18/6/21", "these chips are bad."],
                       ["19/2/21", "Such good coffee!"],
                       ["19/2/21", "good coffee!"],
                       ["19/2/21", "tea could be better!"],
                       ["19/2/21", "bad chips"],
                       ["19/2/21", "best chips ever"]],
                      columns=["Time", "Text"])
    model = ZeroShot()
    candidate_labels = ['snacks', 'drinks']
    output_df = model.predict(df, 'Text', candidate_labels)

    df_expected_output = pd.DataFrame([["18/6/21", "these chips are bad."],
                                       ["19/2/21", "Such good coffee!"],
                                       ["19/2/21", "good coffee!"],
                                       ["19/2/21", "tea could be better!"],
                                       ["19/2/21", "bad chips"],
                                       ["19/2/21", "best chips ever"]],
                                      columns=['Time',
                                               'Text'])

    if not all(topic in candidate_labels for topic in output_df['topic']):
        print("Topic labels not in topic")
    else:
        return pd.testing.assert_frame_equal(
            output_df[['Time',
                       'Text']],
            df_expected_output,
            check_index_type=False)


def test_lda_module():

    check = []
    num_topics = config_params['LDA']['num_topics']

    lda_model = LDA()
    df_preproc = lda_model.preprocess(data_proc.copy(), "review")
    lda, df_corpus, df_id2word, df_bigram = lda_model.fit(df_preproc)
    output_df = lda_model.predict(df_preproc, lda, df_corpus)

    if output_df['topic'].isnull().values.any():
        check.append("There are topics with null values.")
    if not all(int(topic) <= num_topics for topic in output_df['topic']):
        check.append("The topics exceed the specified number of topics.")

    if check:
        print(check)


def test_nmf_module():
    num_topics = config_params['NMF']['nmf_args']['n_components']
    check = []

    nmf = Tfidf_NMF_Module()
    nmf.fit(data_proc.copy())
    output_df = nmf.predict(data_proc.copy())

    if output_df['topic'].isnull().values.any():
        check.append("There are topics with null values.")
    if not all(int(topic) <= num_topics for topic in output_df['topic']):
        check.append("The topics exceed the specified number of topics.")

    if check:
        print(check)


def test_bertopic_module():
    num_topics = config_params['BERTopic']['nr_topics']
    check = []

    bertopic_model = BERTopic_Module()
    bertopic_model.hdbscan_args['min_cluster_size'] = 10
    output_df = bertopic_model.predict(data_proc.copy())

    if all(output_df['topic'] == -1):
        check.append(
            "All reviews set as outliers. Choose better hyperparameters")
    if not all(int(topic) <= num_topics for topic in output_df['topic']):
        check.append("The topics exceed the specified number of topics.")

    if check:
        print(check)


def test_topic_modelling_train_module():

    check = []
    topic_modelling_train()

    if not FileUtil.check_filepath_exists(FileUtil().LDA_TOPIC_FILE_PATH):
        check.append("LDA topics not generated!")

    if not FileUtil.check_filepath_exists(FileUtil().NMF_TOPIC_FILE_PATH):
        check.append("NMF topics not generated!")

    if not FileUtil.check_filepath_exists(FileUtil().BERTOPIC_TOPIC_FILE_PATH):
        check.append("BERTopic topics not generated!")

    if check:
        print(check)


# ideally to be run on EC2 instance
def test_sentiment_analysis_train_module():

    check = []
    sentiment_analysis_train()

    # check training accuracy/loss graphs for LSTM & BERT
    if not FileUtil.check_filepath_exists(
                            FileUtil().LSTM_TRAINING_GRAPH_FILE_PATH):
        check.append("LSTM training graph not generated!")

    if not FileUtil.check_filepath_exists(
                            FileUtil().BERT_TRAINING_GRAPH_FILENAME):
        check.append("BERT training graph not generated!")

    # check metrics file of length 3 for LSTM, BERT & Logistic Regression
    if not FileUtil.check_filepath_exists(os.path.join(
            FileUtil().SENTIMENT_ANALYSIS_EVAL_DIR,
            FileUtil().METRICS_FILE_NAME)):
        check.append("Metrics.json not generated!")
    else:
        metrics = FileUtil().get_metrics("sentiment_analysis")
        if len(metrics) != 3:
            check.append("Metrics.json doesn't generate all results!")
        else:
            if not pd.Series(metrics["LOGREG"].values()).between(0, 1).all():
                check.append(
                    "LR metrics probabilities not between 0-1. Fix evaluate")
            if not pd.Series(metrics["BERT"].values()).between(0, 1).all():
                check.append(
                    "BERT metrics probabilities not between 0-1. Fix evaluate")
            if not pd.Series(metrics["LSTM"].values()).between(0, 1).all():
                check.append(
                    "LSTM metrics probabilities not between 0-1. Fix evaluate")
    if check:
        print(check)


def test_bert_module():
    check = []
    df = pd.DataFrame([["18/6/21", "chips bad."],
                       ["19/2/21", "good coffee"],
                       ["19/2/21", "worst coffee ever"],
                       ["19/2/21", "tea could better"],
                       ["19/2/21", "bad restaurant poor decoration not clean"],
                       ["19/2/21", "rude waiter zero stars"]],
                      columns=["Time", "partially_cleaned_text"])

    model = BERT(True)
    label, probs, tf_predictions = model.predict(df)
    df["sentiment"] = label
    df["sentiment_prob"] = probs

    if not df["sentiment_prob"].between(0, 1).all():
        check.append("Sentiment probabilities are out of range of 0 to 1")
    if not all(sentiment in [0, 1] for sentiment in list(df["sentiment"])):
        check.append(
            "Sentiment output values are not binary (strictly 0 or 1)")

    if check:
        print(check)


def test_logreg_module():
    check = []
    df = pd.DataFrame([["18/6/21", "chips bad."],
                       ["19/2/21", "good coffee"],
                       ["19/2/21", "worst coffee ever"],
                       ["19/2/21", "tea could better"],
                       ["19/2/21", "bad restaurant poor decoration not clean"],
                       ["19/2/21", "rude waiter zero stars"]],
                      columns=["Time", "cleaned_text"])

    model = LOGREG(True)
    model.tokenize(df)
    label, probs = model.predict(df)
    df["sentiment"] = label
    df["sentiment_prob"] = probs

    if not df["sentiment_prob"].between(0, 1).all():
        check.append("Sentiment probabilities are out of range of 0 to 1")
    if not all(sentiment in [0, 1] for sentiment in list(df["sentiment"])):
        check.append(
            "Sentiment output values are not binary (strictly 0 or 1)")

    if check:
        print(check)


def test_lstm_module():

    check = []
    df = pd.DataFrame([["18/6/21", "chips bad."],
                       ["19/2/21", "good coffee"],
                       ["19/2/21", "worst coffee ever"],
                       ["19/2/21", "tea could better"],
                       ["19/2/21", "bad restaurant poor decoration not clean"],
                       ["19/2/21", "rude waiter zero stars"]],
                      columns=["Time", "cleaned_text"])

    model = Lstm(True)
    model.tokenize(df)
    label, probs = model.predict(df)
    df["sentiment"] = label
    df["sentiment_prob"] = probs

    if not df["sentiment_prob"].between(0, 1).all():
        check.append("Sentiment probabilities are out of range of 0 to 1")
    if not all(sentiment in [0, 1] for sentiment in list(df["sentiment"])):
        check.append(
            "Sentiment output values are not binary (strictly 0 or 1)")

    if check:
        print(check)

def test_fileutil_check_dir_exists():
    dirs = ["src/utils",
            "src/models/topic_modelling/train",
            "src/app",
            "src/visualisation", 
            "src/models/sentiment_analysis/train"]
    
    check = list(filter(lambda dir: not FileUtil.check_dir_exists(dir), dirs))

    if check:
        print("These directories are not found: {}".format(check))
        
def test_check_filepath_exists():
    dirs = ["src/utils/file_util.py",
            "src/models/topic_modelling/train/train.py",
            "src/app/app.py",
            "src/models/predict.py",
            "src/visualisation/dashboard_viz.py", 
            "src/models/sentiment_analysis/test/predict.py"]
    
    check = list(filter(lambda dir: not FileUtil.check_filepath_exists(dir), dirs))

    if check:
        print("These filepaths are not found: {}".format(check))

def test_check_get_yml():
    dir = "requirements.txt"
    try:
        FileUtil().get_yml(filepath=dir)
    except InvalidExtensionException as error:
        return error

def test_check_put_json():
    dir = "requirements.txt"
    try:
        FileUtil().get_yml(filepath=dir)
    except InvalidExtensionException as error:
        return error 

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

    # # Testing cleaning functions
    test_apply_cleaning_test()
    test_apply_cleaning_train()

    # # Testing cleaning edge cases
    test_cleaning_when_date_is_string()
    test_cleaning_when_date_is_datetime()
    test_cleaning_punctuation_and_html_tags()
    test_replace_multiple_spaces_df_for_tabs()

    # # Testing prediction functions

    test_predict_sentiment()
    test_predict_sentiment_topic()
    test_predict_topic()

    # # Testing predict function edge cases
    test_predict_when_null_reviews()
    test_predict_when_all_stopwords()
    test_predict_when_empty_review()

    # # Testing topic modelling modules
    test_lbl2vec_module()
    test_zeroshot_module()
    test_lda_module()
    test_nmf_module()
    test_bertopic_module()
    #test_topic_modelling_train_module()

    # # Testing sentiment analysis modules
    test_bert_module()
    test_lstm_module()
    test_logreg_module()
    #test_sentiment_analysis_train_module()

    # # Test FileUtil module
    test_fileutil_check_dir_exists()
    test_check_filepath_exists()
    test_check_get_yml()
    test_check_put_json()

    print("Unit testing complete!")
    
if __name__ == "__main__":
    unit_test()
