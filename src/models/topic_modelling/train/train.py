"""This module contains topic_modelling_train function."""

import plotly

from src.models.topic_modelling.train.lda import LDA
from src.models.topic_modelling.train.bertopic import BERTopic_Module
from src.models.topic_modelling.train.nmf import Tfidf_NMF_Module
from src.utils.file_util import FileUtil


def topic_modelling_train():
    """
    Train topic models on training data.

    LDA, BERTopic, and NMF models will be fitted on training data.
    All 3 models' topics plot will be saved to eval folder.
    """
    df = FileUtil.get_processed_train_data()

    # LDA
    lda_model = LDA()
    df_preproc = lda_model.preprocess(df.copy(), "review")
    lda, df_corpus, df_id2word, df_bigram = lda_model.fit(df_preproc)
    df_pred = lda_model.predict(df_preproc, lda, df_corpus)
    fig = lda_model.evaluate(df_pred)
    FileUtil.put_topics_html("LDA", fig)

    # BERTopic
    bertopic_model = BERTopic_Module()
    df_pred = bertopic_model.predict(df.copy())
    fig = bertopic_model.evaluate(df_pred)
    FileUtil.put_topics_html("BERTopic", fig)

    # NMF
    nmf = Tfidf_NMF_Module()
    nmf.fit(df.copy())
    df_pred = nmf.predict(df.copy())
    fig = nmf.evaluate(df_pred)
    FileUtil.put_topics_html("NMF", fig)


if __name__ == "__main__":
    topic_modelling_train()
