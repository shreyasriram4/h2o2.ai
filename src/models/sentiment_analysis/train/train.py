from sklearn.model_selection import train_test_split
from src.models.sentiment_analysis.train.bert import BERT
from src.models.sentiment_analysis.train.logreg import LOGREG
from src.utils.file_util import FileUtil
import joblib


def main():
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

    logreg_model = LOGREG()

    tokenized_df = logreg_model.tokenize(df)
    train, valid = train_test_split(tokenized_df, test_size=0.2)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    trained_logreg_model = logreg_model.fit(train)
    joblib.dump(trained_logreg_model, FileUtil().LOGREG_SENTIMENT_MODEL_DIR)
    logreg_ap, logreg_pr_auc = logreg_model.evaluate(valid)

    FileUtil.put_metrics("sentiment_analysis",
                         {"BERT": {"PR AUC": bert_pr_auc,
                                   "Average Precision": bert_ap}},
                         {"LOGREG": {"PR AUC": logreg_pr_auc,
                                     "Average Precision": logreg_ap}})


if __name__ == "__main__":
    main()
