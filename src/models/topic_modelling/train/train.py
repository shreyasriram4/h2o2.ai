"""This module contains main function for topic model training."""

import plotly

from src.models.topic_modelling.train.lda import LDA
from src.models.topic_modelling.train.bertopic import BERTopic_Module
from src.models.topic_modelling.train.nmf import Tfidf_NMF_Module
from src.utils.file_util import FileUtil


def main():
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
    FileUtil.create_dir_if_not_exists(FileUtil().TOPIC_MODELLING_EVAL_DIR)
    plotly.offline.plot(fig, filename=FileUtil().LDA_TOPIC_FILE_PATH)

    # BERTopic
    bertopic_model = BERTopic_Module()
    df_pred = bertopic_model.predict(df.copy())
    fig = bertopic_model.evaluate(df_pred)
    plotly.offline.plot(fig, filename=FileUtil().BERTOPIC_TOPIC_FILE_PATH)

    # NMF
    nmf = Tfidf_NMF_Module()
    nmf.fit(df.copy())
    df_pred = nmf.predict(df.copy())
    fig = nmf.evaluate(df_pred)
    plotly.offline.plot(fig, filename=FileUtil().NMF_TOPIC_FILE_PATH)


if __name__ == "__main__":
    main()
