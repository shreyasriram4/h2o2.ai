import pandas as pd

from src.utils.file_util import FileUtil
from src.models.topic_modelling.test.lbl2vec import Lbl2Vec
from src.models.topic_modelling.test.zero_shot import ZeroShot


def predict_topic(df, model_name="Lbl2Vec"):
    assert model_name in ["ZeroShot", "Lbl2Vec"]

    config = FileUtil.get_config()

    if model_name == "ZeroShot":
        candidate_labels = config["topics"]
        df = ZeroShot().predict(df, "cleaned_text", candidate_labels)
    else:
        topic_mapping = config["topic_mapping"]
        candidate_labels = {}
        for topic, subtopics in topic_mapping.items():
            for subtopic in subtopics:
                candidate_labels[subtopic] = topic
        df = Lbl2Vec().predict(df, "cleaned_text", candidate_labels)

    return df
