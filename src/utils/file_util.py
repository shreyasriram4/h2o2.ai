"""This module contains FileUtil and InvalidExtensionException classes,
and _check_filepath function."""

import functools
import json
import os
import pickle
import plotly
import re

import pandas as pd
import yaml


class InvalidExtensionException(Exception):
    """An Exception for invalid filepath extension."""

    pass


def _check_filepath(ext):
    """
    Pre-check whether the filepath ends with ext extension.

    Args:
      ext: extension to check.
    """
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            filepath = kwargs.get("filepath")
            if not filepath:
                filepath = args[1]

            if not filepath.endswith(ext):
                raise InvalidExtensionException(
                    f"{filepath} has invalid extension, want {ext}")

            return f(*args, **kwargs)

        return _wrapper

    return _decorator


class FileUtil():
    """A class to access storage."""

    def __init__(self, config_path="config.yml"):
        """
        Constructor for FileUtil class.

        Args:
          config_path (str, optional): path to config file.
          Default is config.yml
        """
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
        self.LSTM_TRAINING_LOSS_GRAPH_FILENAME = self.CONFIG_PARAMS[
            "lstm_training_loss_graph_filename"]
        self.LSTM_TRAINING_ACC_GRAPH_FILENAME = self.CONFIG_PARAMS[
            "lstm_training_acc_graph_filename"]
        self.LDA_TOPIC_FILE_NAME = "lda_topics.html"
        self.BERTOPIC_TOPIC_FILE_NAME = "bertopic_topics.html"
        self.NMF_TOPIC_FILE_NAME = "nmf_topics.html"

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
        self.SENTIMENT_ANALYSIS_EVAL_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_DIR, "eval")
        self.BERT_SENTIMENT_MODEL_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_TRAIN_DIR, "bert_model")
        self.BERT_TRAINING_GRAPH_FILE_PATH = os.path.join(
            self.SENTIMENT_ANALYSIS_EVAL_DIR,
            self.BERT_TRAINING_GRAPH_FILENAME)

        self.LSTM_SENTIMENT_MODEL_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_TRAIN_DIR, "lstm_model")
        self.LSTM_TRAINING_LOSS_GRAPH_FILE_PATH = os.path.join(
            self.SENTIMENT_ANALYSIS_EVAL_DIR,
            self.LSTM_TRAINING_LOSS_GRAPH_FILENAME)
        self.LSTM_TRAINING_ACC_GRAPH_FILE_PATH = os.path.join(
            self.SENTIMENT_ANALYSIS_EVAL_DIR,
            self.LSTM_TRAINING_ACC_GRAPH_FILENAME)

        self.LOGREG_SENTIMENT_MODEL_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_TRAIN_DIR, "logreg_model/logreg_model.sav")
        self.LOGREG_SENTIMENT_W2V_MODEL_DIR = os.path.join(
            self.SENTIMENT_ANALYSIS_TRAIN_DIR,
            "logreg_model/logreg_word2vec.model")

        self.TOPIC_MODELLING_TRAIN_DIR = os.path.join(self.TOPIC_MODELLING_DIR,
                                                      "train")
        self.TOPIC_MODELLING_EVAL_DIR = os.path.join(self.TOPIC_MODELLING_DIR,
                                                     "eval")
        self.LDA_TOPIC_FILE_PATH = os.path.join(self.TOPIC_MODELLING_EVAL_DIR,
                                                self.LDA_TOPIC_FILE_NAME)
        self.BERTOPIC_TOPIC_FILE_PATH = os.path.join(
            self.TOPIC_MODELLING_EVAL_DIR,
            self.BERTOPIC_TOPIC_FILE_NAME)
        self.NMF_TOPIC_FILE_PATH = os.path.join(self.TOPIC_MODELLING_EVAL_DIR,
                                                self.NMF_TOPIC_FILE_NAME)

    @_check_filepath(".csv")
    def get_csv(self, filepath: str) -> pd.DataFrame:
        """
        Get the csv from filepath.

        Args:
          filepath (str): csv filepath

        Returns:
          dataframe (pd.Dataframe): dataframe of the csv

        Raises:
          InvalidExtensionException: If filepath doesn't have csv extension
        """
        return pd.read_csv(filepath)

    @_check_filepath(".csv")
    def put_csv(self, filepath: str, df: pd.DataFrame) -> None:
        """
        Put the dataframe to csv in filepath.

        Args:
          filepath (str): csv filepath
          df (pd.Dataframe): dataframe of the csv

        Raises:
          InvalidExtensionException: If filepath doesn't have csv extension
          TypeError: If df is not a pandas DataFrame object
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be of type pd.DataFrame, got {type(df)}")
        df.to_csv(filepath, index=False)

    @_check_filepath(".pkl")
    def get_pkl(self, filepath: str):
        """
        Get the pickle file from filepath.

        Args:
          filepath (str): pickle filepath

        Returns:
          pickle file

        Raises:
          InvalidExtensionException: If filepath doesn't have pkl extension
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @_check_filepath(".pkl")
    def put_pkl(self, filepath: str, python_object) -> None:
        """
        Put the pickle to filepath.

        Args:
          filepath (str): pickle filepath
          python_object (Object): Python object of any type

        Raises:
          InvalidExtensionException: If filepath doesn't have pkl extension
          TypeError: If df is zero or empty or None
        """
        if not python_object:
            raise TypeError(
                "python_object must be non-zero, non-empty, and not None")
        with open(filepath, "wb") as f:
            pickle.dump(python_object, f)

    @_check_filepath(".yml")
    def get_yml(self, filepath: str):
        """
        Get the yaml file from filepath.

        Args:
          filepath (str): yaml filepath

        Returns:
          yaml file

        Raises:
          InvalidExtensionException: If filepath doesn't have yml extension
        """
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    @_check_filepath(".json")
    def put_json(self, filepath: str, dic) -> None:
        """
        Put the json to filepath.

        Args:
          filepath (str): json filepath
          python_object (dict): Python dictionary

        Raises:
          InvalidExtensionException: If filepath doesn't have json extension
          TypeError: If dic is not dictionary
        """
        if not isinstance(dic, dict):
            raise TypeError(f"dic must be of type dict, got {type(dic)}")
        with open(filepath, "w") as f:
            json.dump(dic, f)

    @_check_filepath(".json")
    def get_json(self, filepath: str):
        """
        Get the json from filepath.

        Args:
          filepath (str): json filepath

        Raises:
          InvalidExtensionException: If filepath doesn't have json extension
        """
        with open(filepath) as json_file:
            return json.load(json_file)

    @classmethod
    def check_dir_exists(self, dir):
        """
        Check if directory exists.

        Args:
          dir (str): directory to check

        Returns:
          True if directory exists, False otherwise
        """
        return os.path.exists(dir)

    @classmethod
    def create_dir_if_not_exists(self, dir):
        """
        Create directory if it doesn't exist.

        Args:
          dir (str): directory to create
        """
        if not FileUtil.check_dir_exists(dir):
            os.makedirs(dir)

    @classmethod
    def check_filepath_exists(self, filepath):
        """
        Check if file path exists.

        Args:
          filepath (str): file path to check

        Returns:
          True if file path exists, False otherwise
        """
        return os.path.isfile(filepath)

    @classmethod
    def get_raw_train_data(self) -> pd.DataFrame:
        """
        Get raw train data.

        Returns:
          pandas dataframe of the raw train data
        """
        filepath = os.path.join(FileUtil().RAW_DATA_DIR,
                                FileUtil().TRAIN_FILE_NAME)
        return self.get_csv(self, filepath)

    @classmethod
    def get_processed_train_data(self) -> pd.DataFrame:
        """
        Get processed train data.

        Returns:
          pandas dataframe of the processed train data
        """
        filepath = os.path.join(
            FileUtil().PROCESSED_DATA_DIR, FileUtil().TRAIN_FILE_NAME)
        return self.get_csv(self, filepath)

    @classmethod
    def put_processed_train_data(self, df: pd.DataFrame) -> None:
        """
        Get processed train data.

        Returns:
          pandas dataframe of the processed train data
        """
        FileUtil.create_dir_if_not_exists(FileUtil().PROCESSED_DATA_DIR)
        filepath = os.path.join(
            FileUtil().PROCESSED_DATA_DIR, FileUtil().TRAIN_FILE_NAME)
        self.put_csv(self, filepath, df)

    @classmethod
    def get_topic_model(self):
        """
        Get topic model from a pickle file.

        Returns:
          pickle file of the model
        """
        filepath = os.path.join(
            FileUtil().TOPIC_MODELLING_DIR, FileUtil().MODEL_FILE_NAME)
        return self.get_pkl(self, filepath)

    @classmethod
    def put_topic_model(self, model) -> None:
        """
        Put topic model into a pickle file.

        Args:
          model: topic model to save
        """
        FileUtil.create_dir_if_not_exists(FileUtil().TOPIC_MODELLING_DIR)
        filepath = os.path.join(
            FileUtil().TOPIC_MODELLING_DIR, FileUtil().MODEL_FILE_NAME)
        self.put_pkl(self, filepath, model)

    @classmethod
    def get_config(self):
        """
        Get config file.

        Returns:
          dictionary of config
        """
        filepath = os.path.join(FileUtil().PROJECT_DIR,
                                FileUtil().CONFIG_FILE_NAME)
        return self.get_yml(self, filepath)

    @classmethod
    def put_predicted_df(self, df: pd.DataFrame, filename: str) -> None:
        """
        Put predicted df into a csv file.

        Args:
          df (pd.DataFrame): dataframe to save
          filename (str): csv file name
        """
        FileUtil.create_dir_if_not_exists(FileUtil().PREDICTED_DATA_DIR)
        filepath = os.path.join(FileUtil().PREDICTED_DATA_DIR, filename)
        self.put_csv(self, filepath, df)

    @classmethod
    def put_metrics(self, task: str, dic) -> None:
        """
        Put metrics into json file.

        Args:
          task (str): VOC task
          dic (dict): dictionary of metrics
        """
        if task == "sentiment_analysis":
            FileUtil.create_dir_if_not_exists(
                FileUtil().SENTIMENT_ANALYSIS_EVAL_DIR)
            filepath = os.path.join(
                FileUtil().SENTIMENT_ANALYSIS_EVAL_DIR,
                FileUtil().METRICS_FILE_NAME)
            self.put_json(self, filepath, dic)

    @classmethod
    def get_metrics(self, task: str):
        """
        Get metrics from json file.

        Args:
          task (str): VOC task
        """
        if task == "sentiment_analysis":
            filepath = os.path.join(
                FileUtil().SENTIMENT_ANALYSIS_EVAL_DIR,
                FileUtil().METRICS_FILE_NAME)
            return self.get_json(self, filepath)

    @classmethod
    def put_topics_html(self, model_name: str, fig) -> None:
        """
        Put topics html into topic modelling eval folder

        Args:
          model_name (str): model name of the topics
          fig: Plotly figure
        """
        assert model_name in ["LDA", "BERTopic", "NMF"]

        FileUtil.create_dir_if_not_exists(FileUtil().TOPIC_MODELLING_EVAL_DIR)

        if model_name == "LDA":
            plotly.offline.plot(fig, filename=FileUtil().LDA_TOPIC_FILE_PATH)
        elif model_name == "BERTopic":
            plotly.offline.plot(fig,
                                filename=FileUtil().BERTOPIC_TOPIC_FILE_PATH)
        else:
            plotly.offline.plot(fig, filename=FileUtil().NMF_TOPIC_FILE_PATH)

    @classmethod
    def get_topics_html(self, model_name: str) -> None:
        """
        Get topics html from topic modelling eval folder

        Args:
          model_name (str): model name of the topics

        Returns:
          Plotly figure of the topics
        """
        assert model_name in ["LDA", "BERTopic", "NMF"]

        FileUtil.create_dir_if_not_exists(FileUtil().TOPIC_MODELLING_EVAL_DIR)

        if model_name == "LDA":
            filepath = FileUtil().LDA_TOPIC_FILE_PATH
        elif model_name == "BERTopic":
            filepath = FileUtil().BERTOPIC_TOPIC_FILE_PATH
        else:
            filepath = FileUtil().NMF_TOPIC_FILE_PATH

        with open(filepath, encoding="utf8") as f:
            html = f.read()

        call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[-2**16:])[0]
        call_args = json.loads(f'[{call_arg_str}]')
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        return plotly.io.from_json(json.dumps(plotly_json))
