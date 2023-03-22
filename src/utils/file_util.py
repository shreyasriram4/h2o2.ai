import functools
import os
import pandas as pd
import pickle

class InvalidExtension(Exception):
    pass

def _check_filepath(ext):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            filepath = kwargs.get("filepath")
            if not filepath:
                filepath = args[1]

            if not filepath.endswith(ext):
                raise InvalidExtension(f"{filepath} has invalid extension, want {ext}")

            return f(*args, **kwargs)

        return _wrapper

    return _decorator

class FileUtil():
    TRAIN_FILE_NAME = "reviews.csv"
    MODEL_FILE_NAME = "model.pkl"

    PROJECT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )

    RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data/raw")
    PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed")

    SENTIMENT_ANALYSIS_DIR = os.path.join(PROJECT_DIR, "src/models/sentiment_analysis")
    TOPIC_MODELLING_DIR = os.path.join(PROJECT_DIR, "src/models/topic_modelling")

    @_check_filepath(".csv")
    def get_csv(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    @_check_filepath(".csv")
    def put_csv(self, filepath: str, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be of type pd.DataFrame, got {type(df)}")
        df.to_csv(filepath, index=False)

    @_check_filepath(".pkl")
    def get_pkl(self, filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @_check_filepath(".pkl")
    def put_pkl(self, filepath: str, python_object) -> None:
        if not python_object:
            raise TypeError("python_object must be non-zero, non-empty, and not None")
        with open(filepath, "wb") as f:
            pickle.dump(python_object, f)

    def create_dir_if_not_exists(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    @classmethod
    def get_raw_train_data(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(FileUtil.RAW_DATA_DIR, filename)
        return self.get_csv(filepath)

    @classmethod
    def get_processed_train_data(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(FileUtil.PROCESSED_DATA_DIR, filename)
        return self.get_csv(filepath)

    @classmethod
    def put_processed_train_data(self, filename: str, df: pd.DataFrame) -> None:
        self.create_dir_if_not_exists(FileUtil.PROCESSED_DATA_DIR)
        filepath = os.path.join(FileUtil.PROCESSED_DATA_DIR, filename)
        self.put_csv(filepath, df)

    @classmethod
    def get_topic_model(self):
        filepath = os.path.join(FileUtil.TOPIC_MODELLING_DIR, FileUtil.MODEL_FILE_NAME)
        return self.get_pkl(filepath)

    @classmethod
    def put_topic_model(self, model) -> None:
        self.create_dir_if_not_exists(FileUtil.TOPIC_MODELLING_DIR)
        filepath = os.path.join(FileUtil.TOPIC_MODELLING_DIR, FileUtil.MODEL_FILE_NAME)
        self.put_pkl(filepath, model)

    @classmethod
    def get_sentiment_model(self):
        filepath = os.path.join(FileUtil.SENTIMENT_ANALYSIS_DIR, FileUtil.MODEL_FILE_NAME)
        return self.get_pkl(filepath)

    @classmethod
    def put_sentiment_model(self, model) -> None:
        self.create_dir_if_not_exists(FileUtil.SENTIMENT_ANALYSIS_DIR)
        filepath = os.path.join(FileUtil.SENTIMENT_ANALYSIS_DIR, FileUtil.MODEL_FILE_NAME)
        self.put_pkl(filepath, model)