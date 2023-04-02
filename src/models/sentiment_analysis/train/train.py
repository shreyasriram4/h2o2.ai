"""This module contains main function for training."""

from sklearn.model_selection import train_test_split
from src.models.sentiment_analysis.train.bert import BERT
from src.utils.file_util import FileUtil


def main():
    """
    Train sentiment analysis models on training data.

    Processed dataset will be split into 80% train and 20% valid.
    BERT, Logistic Regression, and LSTM model will be fitted on
    training data and evaluated on validation data.
    All 3 models and metrics, and BERT training graph will be saved to storage.
    """
    df = FileUtil.get_processed_train_data()
    train, valid = train_test_split(df, test_size=0.2)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    bert_model = BERT()
    trained_bert_model, history = bert_model.fit(train.copy(), valid.copy())
    trained_bert_model.save_pretrained(FileUtil().BERT_SENTIMENT_MODEL_DIR)
    bert_ap, bert_pr_auc = bert_model.evaluate(valid.copy())
    bert_model.plot_training_acc_loss(history)

    #  LogReg and LSTM training then add to the metrics below

    FileUtil.put_metrics("sentiment_analysis",
                         {"BERT": {"PR AUC": bert_pr_auc,
                                   "Average Precision": bert_ap}})


if __name__ == "__main__":
    main()
