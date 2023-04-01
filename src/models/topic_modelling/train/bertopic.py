import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from hdbscan import HDBSCAN

from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
from src.visualisation.visualise_topics import visualise_top_words
from src.preprocessing.transformations import apply_cleaning_train


class BERTopic_Module(Classifier):
    def __init__(self):
        self.config_params = FileUtil.get_config()
        self.custom_stopwords = self.config_params['custom_stopwords']
        self.bertopic_config = self.config_params["BERTopic"]
        self.nr_topics = self.bertopic_config['nr_topics']

        if 'vectorizer_model' in self.bertopic_config.keys():
            if self.bertopic_config["vectorizer_model"] in (
                ['CountVectorizer',
                 'TfidfVectorizer']
            ):
                self.vectorizer_model = self.bertopic_config[
                    'vectorizer_model'
                ]
                if 'vectorizer_args' in self.bertopic_config.keys():
                    self.vectorizer_args = self.bertopic_config[
                        'vectorizer_args'
                    ]

        if 'hdbscan_args' in self.bertopic_config.keys():
            self.hdbscan_args = self.bertopic_config['hdbscan_args']

    #def preprocessing(self, df):
        #df = apply_cleaning_train(df)
    #    return df

    def fit(self):
        pass

    def predict(self, df):
        bertopic_args = {}
        bertopic_args['nr_topics'] = self.nr_topics

        #df = self.preprocessing(df)

        if self.vectorizer_model:
            if self.vectorizer_model == 'CountVectorizer':
                vectorizer_model = CountVectorizer(**self.vectorizer_args)
            elif self.vectorizer_model == 'TfidfVectorizer':
                vectorizer_model = TfidfVectorizer(**self.vectorizer_args)
            bertopic_args["vectorizer_model"] = vectorizer_model

        if self.hdbscan_args:
            hdbscan_model = HDBSCAN(**self.hdbscan_args)
            bertopic_args["hdbscan_model"] = hdbscan_model

        model = BERTopic(**bertopic_args)
        topics, probs = model.fit_transform(df["partially_cleaned_text"])

        df["topic"] = topics

        return df

    def evaluate(self, df):
        topics = list(set(df["topic"]))
        topics = [topic_num for topic_num in topics if topic_num != -1]
        fig = visualise_top_words(
            df, topics,
            custom_sw=self.custom_stopwords,
            inc_size=True
        )
        return fig
