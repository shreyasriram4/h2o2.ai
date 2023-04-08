"""This module contains sentiment_analysis_train function."""

from sklearn.model_selection import train_test_split
from src.models.sentiment_analysis.train.bert import BERT
from src.models.sentiment_analysis.train.lstm import Lstm
from src.models.sentiment_analysis.train.logreg import LOGREG
from src.utils.file_util import FileUtil
import joblib


def sentiment_analysis_train():
    """
    Train sentiment analysis models on training data.

    Processed dataset will be split into 80% train and 20% valid.
    BERT, Logistic Regression, and LSTM models will be fitted on
    training data and evaluated on validation data.
    All 3 models and metrics, and BERT training graph will be saved to storage.
    """
    df = FileUtil.get_processed_train_data()
    train, valid = train_test_split(df, test_size=0.2, random_state=1)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    bert_model = BERT()
    trained_bert_model, history = bert_model.fit(train.copy(), valid.copy())
    trained_bert_model.save_pretrained(FileUtil().BERT_SENTIMENT_MODEL_DIR)
    bert_ap, bert_pr_auc = bert_model.evaluate(valid.copy())
    bert_model.plot_training_acc_loss(history)

    lstm_model = Lstm()
    df_lstm = df.copy()
    tokenized_df = lstm_model.tokenize(df_lstm)
    train, valid = train_test_split(
        tokenized_df, test_size=0.2, random_state=1)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    w2v_model = lstm_model.train_w2v_model(tokenized_df)
    lstm_model.get_embedding_matrix()

    lstm_model.build_model()
    lstm_model.fit(train, valid)
    lstm_model.plot_training_metrics()
    lstm_ap, lstm_pr_auc = lstm_model.evaluate(valid)

    logreg_model = LOGREG()
    df_logreg = df.copy()
    tokenized_df = logreg_model.tokenize(df_logreg)
    train, valid = train_test_split(
        tokenized_df, test_size=0.2, random_state=1)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    logreg_model.train_w2v_model(train)
    trained_logreg_model = logreg_model.fit(train)
    joblib.dump(trained_logreg_model, FileUtil().LOGREG_SENTIMENT_MODEL_PATH)
    logreg_ap, logreg_pr_auc = logreg_model.evaluate(valid)

    FileUtil.put_metrics("sentiment_analysis",
                         {"BERT": {"PR AUC": bert_pr_auc,
                                   "Average Precision": bert_ap},
                          "LSTM": {"PR AUC": lstm_pr_auc,
                                   "Average Precision": lstm_ap},
                          "LOGREG": {"PR AUC": logreg_pr_auc,
                                     "Average Precision": logreg_ap}})


if __name__ == "__main__":
    sentiment_analysis_train()
