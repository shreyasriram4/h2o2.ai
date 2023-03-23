import functools
import json
import os
import pandas as pd
import pickle
import yaml

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
    CONFIG_FILE_NAME = "config.yml"
    METRICS_FILE_NAME = "metrics.json"

    PROJECT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )

    RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data/raw")
    PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed")
    PREDICTED_DATA_DIR = os.path.join(PROJECT_DIR, "data/predicted")

    SENTIMENT_ANALYSIS_DIR = os.path.join(PROJECT_DIR, "src/models/sentiment_analysis")
    TOPIC_MODELLING_DIR = os.path.join(PROJECT_DIR, "src/models/topic_modelling")

    BEST_SENTIMENT_MODEL_DIR = os.path.join(SENTIMENT_ANALYSIS_DIR, "train/best_model")

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

    @_check_filepath(".yml")
    def get_yml(self, filepath: str):
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
        
    @_check_filepath(".json")
    def put_json(self, filepath: str, dic) -> None:
        if not isinstance(dic, dict):
            raise TypeError(f"dic must be of type dict, got {type(dic)}")
        with open(filepath, "w") as f:
            json.dump(dic, f)
        
    @classmethod
    def check_dir_exists(self, dir):
        return os.path.exists(dir)

    @classmethod
    def create_dir_if_not_exists(self, dir):
        if not FileUtil.check_dir_exists(dir):
            os.makedirs(dir)

    @classmethod
    def check_filepath_exists(self, filepath):
        return os.path.isfile(filepath)

    @classmethod
    def get_raw_train_data(self) -> pd.DataFrame:
        filepath = os.path.join(FileUtil.RAW_DATA_DIR, FileUtil.TRAIN_FILE_NAME)
        return self.get_csv(self, filepath)

    @classmethod
    def get_processed_train_data(self) -> pd.DataFrame:
        filepath = os.path.join(FileUtil.PROCESSED_DATA_DIR, FileUtil.TRAIN_FILE_NAME)
        return self.get_csv(self, filepath)

    @classmethod
    def put_processed_train_data(self, df: pd.DataFrame) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil.PROCESSED_DATA_DIR)
        filepath = os.path.join(FileUtil.PROCESSED_DATA_DIR, FileUtil.TRAIN_FILE_NAME)
        self.put_csv(self, filepath, df)

    @classmethod
    def get_topic_model(self):
        filepath = os.path.join(FileUtil.TOPIC_MODELLING_DIR, FileUtil.MODEL_FILE_NAME)
        return self.get_pkl(self, filepath)

    @classmethod
    def put_topic_model(self, model) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil.TOPIC_MODELLING_DIR)
        filepath = os.path.join(FileUtil.TOPIC_MODELLING_DIR, FileUtil.MODEL_FILE_NAME)
        self.put_pkl(self, filepath, model)

    @classmethod
    def get_sentiment_model(self):
        filepath = os.path.join(FileUtil.SENTIMENT_ANALYSIS_DIR, FileUtil.MODEL_FILE_NAME)
        return self.get_pkl(self, filepath)

    @classmethod
    def put_sentiment_model(self, model) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil.SENTIMENT_ANALYSIS_DIR)
        filepath = os.path.join(FileUtil.SENTIMENT_ANALYSIS_DIR, FileUtil.MODEL_FILE_NAME)
        self.put_pkl(self, filepath, model)

    @classmethod
    def get_config(self):
        filepath = os.path.join(FileUtil.PROJECT_DIR, FileUtil.CONFIG_FILE_NAME)
        return self.get_yml(self, filepath)
    
    @classmethod
    def put_predicted_df(self, df: pd.DataFrame, filename: str) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil.PREDICTED_DATA_DIR)
        filepath = os.path.join(FileUtil.PREDICTED_DATA_DIR, filename)
        self.put_csv(self, filepath, df)

    @classmethod
    def put_metrics(self, task: str, dic) -> None:
        if task == "sentiment_analysis":
            filepath = os.path.join(FileUtil.SENTIMENT_ANALYSIS_DIR, FileUtil.METRICS_FILE_NAME)
        self.put_json(self, filepath, dic)
