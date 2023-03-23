import pandas as pd
from src.utils.file_util import FileUtil
from src.models.topic_modelling.test.lbl2vec import Lbl2Vec
from src.models.topic_modelling.test.zero_shot import ZeroShot

def predict_topic(df, model_name = "Lbl2Vec"):
    assert model_name in ["ZeroShot", "Lbl2Vec"]

    topic_dict = FileUtil.get_topics()

    if model_name == "ZeroShot":
        candidate_labels = topic_dict["topics"]
        df = ZeroShot().predict(df, "cleaned_text", candidate_labels)
    else:
        candidate_labels = topic_dict["topic_mapping"]
        df = Lbl2Vec().predict(df, "cleaned_text", candidate_labels)

    return df