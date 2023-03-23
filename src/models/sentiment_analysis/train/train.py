from sklearn.model_selection import train_test_split
from src.models.sentiment_analysis.train.bert import BERT
from src.utils.file_util import FileUtil

def main():
    df = FileUtil.get_processed_train_data()
    train, valid = train_test_split(df, test_size = 0.2)

    train = train.reset_index(drop = True)
    valid = valid.reset_index(drop = True)

    model = BERT()

    history = model.fit(train, valid)

    pr_auc = model.evaluate(valid)

    model.plot_training_acc_loss(history)

if __name__ == "__main__":
    main()