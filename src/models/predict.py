import pandas as pd
from datetime import datetime
from src.utils.file_util import FileUtil
from src.models.sentiment_analysis.test.predict import predict_sentiment
from src.models.topic_modelling.test.predict import predict_topic
from src.preprocessing.transformations import apply_cleaning

def predict_sentiment_topic(file_path, df = FileUtil.get_raw_train_data()):
    if file_path:
        df = pd.read_csv(file_path)
    
    df = apply_cleaning(df)

    df = df.iloc[:10,] #to be removed

    df = predict_sentiment(df)
    df = predict_topic(df)

    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = dt_string + ".csv"
    FileUtil.put_predicted_df(df, file_name)

    return df
    
if __name__ == "__main__":
    predict_sentiment_topic("")