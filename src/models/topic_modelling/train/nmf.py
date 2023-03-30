import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
from src.visualisation.visualise_topics import visualise_top_words

class Tfidf_NMF(Classifier):
    
    def __init__(self):
        self.config_params = FileUtil.get_config()
        self.nmf_config = self.config_params["NMF"]
        self.custom_stopwords = self.config_params["custom_stopwords"]

        self.vectorizer_args = self.nmf_config['vectorizer_args']
        self.nmf_args = self.nmf_config['nmf_args']
        self.num_topics = self.nmf_args["n_components"]

    def tokenize_df(self, df):
        return df["cleaned_text"].str.split()
    
    def get_corpus(self, df):
        tokenized_corpus = self.tokenize_df(df)
        return tokenized_corpus
    
    def get_n_topics(self, tfidf):
        nmf = NMF(**self.nmf_args).fit(tfidf)

        return nmf
    
    def get_top_n_words(self, nmf_df, tfidf_vectorizer, n):
        components_df = pd.DataFrame(nmf_df.components_, columns=tfidf_vectorizer.get_feature_names_out())
        topic_value = components_df.reset_index().rename(columns={'index':'topic'})
        topic_value = pd.melt(topic_value, id_vars='topic', var_name='word', value_name='score')
        topic_value = topic_value.sort_values(['topic', 'score'], ascending=[True, False]).groupby('topic').head(n)
        return topic_value
    
    def fit(self, df):
        corpus = self.get_corpus(df)
        self.vectorizer_model = TfidfVectorizer(preprocessor = ' '.join, **self.vectorizer_args)
        tfidf = self.vectorizer_model.fit_transform(corpus)
        nmf = self.get_n_topics(tfidf)
        self.nmf = nmf.fit(tfidf)

        return nmf
    
    def predict(self):
        self.top_n_words_df = self.get_top_n_words(self.nmf, self.vectorizer_model, 6)
        return self.top_n_words_df

    def evaluate(self, df):
        topics = list(df["topic"].unique())
        fig = visualise_top_words(
            self.top_n_words_df, topics,
            custom_sw=self.custom_stopwords
            )
        return fig