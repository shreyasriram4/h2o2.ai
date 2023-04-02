import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
from src.visualisation.visualise_topics import visualise_top_words


class Tfidf_NMF_Module(Classifier):

    def __init__(self):
        self.config_params = FileUtil.get_config()
        self.nmf_config = self.config_params["NMF"]
        self.custom_stopwords = self.config_params["custom_stopwords"]

        self.vectorizer_args = self.nmf_config['vectorizer_args']
        self.nmf_args = self.nmf_config['nmf_args']
        self.num_topics = self.nmf_args["n_components"]

    def tokenize_df(self, df):
        df['tokenized_text'] = df["cleaned_text"].str.split()
        return df

    def fit(self, df):

        df = self.tokenize_df(df)

        tfidf_vectorizer = TfidfVectorizer(
            preprocessor=' '.join, **self.vectorizer_args)

        tfidf = tfidf_vectorizer.fit_transform(df["tokenized_text"])

        nmf = NMF(**self.nmf_args).fit(tfidf)

        self.nmf = nmf
        self.tfidf_vectorizer = tfidf_vectorizer

        return tfidf, nmf

    def predict(self, df):

        nmf = self.nmf
        tfidf_vectorizer = self.tfidf_vectorizer

        # Transform the TF-IDF: review_topics matrix

        df = self.tokenize_df(df)

        X = tfidf_vectorizer.transform(df["tokenized_text"])

        review_topics = nmf.transform(X)

        # assigning topics to reviews in df
        topics = pd.DataFrame(review_topics).idxmax(axis=1).astype('string')
        df['topic'] = topics

        return df

    def evaluate(self, df):
        topics = list(set(df["topic"]))
        topics.sort()
        fig = visualise_top_words(
            df, topics,
            custom_sw=self.custom_stopwords
        )
        return fig
