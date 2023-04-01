import pandas as pd
from datetime import datetime
import warnings

from src.utils.file_util import FileUtil
from src.models.sentiment_analysis.test.predict import predict_sentiment
from src.models.topic_modelling.test.predict import predict_topic
from src.preprocessing.transformations import apply_cleaning_test


def predict_sentiment_topic(test_filepath=FileUtil().TEST_FILE_NAME,
                            df=FileUtil.get_raw_train_data()):
    if test_filepath:
        df = pd.read_csv(test_filepath)

    df = apply_cleaning_test(df)

    if len(df) == 0:
        warnings.warn(
            "No entries after cleaning. Returning empty dataframe.")

    # df = df.iloc[:10,] #uncomment to predict just the first 10 rows
    else:
        df = predict_sentiment(df)
        df = predict_topic(df)

    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = dt_string + ".csv"
    FileUtil.put_predicted_df(df, file_name)

    return df


if __name__ == "__main__":
    predict_sentiment_topic()
