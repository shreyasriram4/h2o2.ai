import plotly
from src.models.topic_modelling.train.lda import LDA
from src.utils.file_util import FileUtil


def main():
    df = FileUtil.get_processed_train_data()

    lda_model = LDA()
    df = lda_model.preprocess(df, "review")
    lda, df_corpus, df_id2word, df_bigram = lda_model.fit(df)
    df = lda_model.predict(df, lda, df_corpus)
    fig = lda_model.evaluate(df)
    plotly.offline.plot(fig, filename=FileUtil.LDA_TOPIC_FILE_PATH)

    #  Add NMF and BERTopic


if __name__ == "__main__":
    main()
