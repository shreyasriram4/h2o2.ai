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
                raise InvalidExtension(
                    f"{filepath} has invalid extension, want {ext}")

            return f(*args, **kwargs)

        return _wrapper

    return _decorator


class FileUtil():

    def __init__(self, config_path="config.yml"):

        self.PROJECT_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )

        self.CONFIG_FILE_NAME = config_path

        self.CONFIG_PARAMS = self.get_yml(self.CONFIG_FILE_NAME)

        self.TEST_FILE_NAME = self.CONFIG_PARAMS["test_filename"]
        self.TRAIN_FILE_NAME = self.CONFIG_PARAMS["train_filename"]
        self.MODEL_FILE_NAME = self.CONFIG_PARAMS["model_filename"]
        self.METRICS_FILE_NAME = self.CONFIG_PARAMS["metrics_filename"]
        self.BERT_TRAINING_GRAPH_FILENAME = self.CONFIG_PARAMS[
                                        "bert_training_graph_filename"]
        self.LDA_TOPIC_FILE_NAME = "lda_topics.html"

        self.RAW_DATA_DIR = os.path.join(
            self.PROJECT_DIR, self.CONFIG_PARAMS["raw_data_path"])
        self.PROCESSED_DATA_DIR = os.path.join(
            self.PROJECT_DIR, self.CONFIG_PARAMS["processed_data_path"])
        self.PREDICTED_DATA_DIR = os.path.join(
            self.PROJECT_DIR, self.CONFIG_PARAMS["output_path"])

        self.SENTIMENT_ANALYSIS_DIR = os.path.join(
            self.PROJECT_DIR, self.CONFIG_PARAMS["sentiment_analysis_path"])
        self.TOPIC_MODELLING_DIR = os.path.join(
            self.PROJECT_DIR, self.CONFIG_PARAMS["topic_modelling_path"])

        self.SENTIMENT_ANALYSIS_TRAIN_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_DIR, "train")
        self.BERT_SENTIMENT_MODEL_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_TRAIN_DIR, "bert_model")
        self.BERT_TRAINING_GRAPH_FILE_PATH = os.path.join(
            self.SENTIMENT_ANALYSIS_TRAIN_DIR, 
            self.BERT_TRAINING_GRAPH_FILENAME)
        self.SENTIMENT_ANALYSIS_DIR = os.path.join(
                                            self.PROJECT_DIR,
                                            "src/models/sentiment_analysis")
        self.TOPIC_MODELLING_DIR = os.path.join(self.PROJECT_DIR,
                                                "src/models/topic_modelling")
        self.TOPIC_MODELLING_TRAIN_DIR = os.path.join(self.TOPIC_MODELLING_DIR,
                                                      "train")
        self.LDA_TOPIC_FILE_PATH = os.path.join(self.TOPIC_MODELLING_TRAIN_DIR,
                                                self.LDA_TOPIC_FILE_NAME)

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
            raise TypeError(
                "python_object must be non-zero, non-empty, and not None")
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
        filepath = os.path.join(FileUtil().RAW_DATA_DIR,
                                FileUtil().TRAIN_FILE_NAME)
        return self.get_csv(self, filepath)

    @classmethod
    def get_processed_train_data(self) -> pd.DataFrame:
        filepath = os.path.join(
            FileUtil().PROCESSED_DATA_DIR, FileUtil().TRAIN_FILE_NAME)
        return self.get_csv(self, filepath)

    @classmethod
    def put_processed_train_data(self, df: pd.DataFrame) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil().PROCESSED_DATA_DIR)
        filepath = os.path.join(
            FileUtil().PROCESSED_DATA_DIR, FileUtil().TRAIN_FILE_NAME)
        self.put_csv(self, filepath, df)

    @classmethod
    def get_raw_test_data(self) -> pd.DataFrame:
        if self.TEST_FILE_NAME != "":
            filepath = os.path.join(
                FileUtil().RAW_DATA_DIR, FileUtil().TEST_FILE_NAME)
        return filepath

    @classmethod
    def get_topic_model(self):
        filepath = os.path.join(
            FileUtil().TOPIC_MODELLING_DIR, FileUtil().MODEL_FILE_NAME)
        return self.get_pkl(self, filepath)

    @classmethod
    def put_topic_model(self, model) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil().TOPIC_MODELLING_DIR)
        filepath = os.path.join(
            FileUtil().TOPIC_MODELLING_DIR, FileUtil().MODEL_FILE_NAME)
        self.put_pkl(self, filepath, model)

    @classmethod
    def get_sentiment_model(self):
        filepath = os.path.join(
            FileUtil().SENTIMENT_ANALYSIS_DIR, FileUtil().MODEL_FILE_NAME)
        return self.get_pkl(self, filepath)

    @classmethod
    def put_sentiment_model(self, model) -> None:
        FileUtil.create_dir_if_not_exists(self.SENTIMENT_ANALYSIS_DIR)
        filepath = os.path.join(
            FileUtil().SENTIMENT_ANALYSIS_DIR, FileUtil().MODEL_FILE_NAME)
        self.put_pkl(self, filepath, model)

    @classmethod
    def get_config(self):
        filepath = os.path.join(FileUtil().PROJECT_DIR,
                                FileUtil().CONFIG_FILE_NAME)
        return self.get_yml(self, filepath)

    @classmethod
    def put_predicted_df(self, df: pd.DataFrame, filename: str) -> None:
        FileUtil.create_dir_if_not_exists(FileUtil().PREDICTED_DATA_DIR)
        filepath = os.path.join(FileUtil().PREDICTED_DATA_DIR, filename)
        self.put_csv(self, filepath, df)

    @classmethod
    def put_metrics(self, task: str, dic) -> None:
        if task == "sentiment_analysis":
            filepath = os.path.join(
                FileUtil().SENTIMENT_ANALYSIS_TRAIN_DIR,
                FileUtil().METRICS_FILE_NAME)
        self.put_json(self, filepath, dic)
