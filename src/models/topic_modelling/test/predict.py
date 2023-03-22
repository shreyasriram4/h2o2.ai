import pandas as pd
from src.utils.file_util import FileUtil
from src.models.topic_modelling.test.zero_shot import ZeroShot

def dataloader(df):
    for i in range(len(df)):
        yield df.loc[i, "cleaned_text"]

def predict_topic(df):
    topic_dict = FileUtil.get_topics()

    candidate_labels = topic_dict["topics"]
    clf = ZeroShot().get_model()

    hypothesis_template = "The topic of this review is {}."

    single_topic_prediction = clf(
        dataloader(df),
        candidate_labels, 
        hypothesis_template = hypothesis_template
        )

    single_topic_prediction = pd.DataFrame(single_topic_prediction)

    df['topic'] = single_topic_prediction['labels'].apply(lambda x: x[0])

    return df