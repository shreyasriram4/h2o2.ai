import gensim
import nltk
import pandas as pd
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
from src.visualisation.visualise_topics import visualise_top_words

nltk.download('wordnet')


class LDA(Classifier):
    def __init__(self):
        self.config_params = FileUtil.get_config()
        self.lda_config = self.config_params["LDA"]
        self.num_topics = self.lda_config["num_topics"]
        self.ngram = self.lda_config["ngram"]
        self.bi_min = self.lda_config["bi_min"]
        self.no_below = self.lda_config["no_below"]
        self.no_above = self.lda_config["no_above"]
        self.min_prob = self.lda_config["min_prob"]
        self.common_words = self.lda_config["common_words"]

    def preprocess(self, df, column):
        df[column] = df["cleaned_text"].apply(self.lemmatize)
        df[column] = df[column].apply(self.generate_bigrams)
        df[column] = df[column].apply(self.remove_common_words)

        return df

    def lemmatize(self, text):
        return " ".join(list(map(WordNetLemmatizer().lemmatize,
                                 text.split(" "))))

    def remove_common_words(self, text):
        return " ".join(list(filter(lambda x: x not in self.common_words,
                                    text.split(" "))))

    def generate_bigrams(self, text):
        result = text
        for w in ngrams(text.split(" "), self.ngram):
            result += " " + "_".join(w)
        return result

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def bigrams(self, words):
        bigram = gensim.models.Phrases(words, min_count=self.bi_min)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod

    def get_corpus(self, df, column):
        words = list(self.sent_to_words(df[column]))
        bigram_mod = self.bigrams(words)
        bigram = [bigram_mod[word] for word in words]
        id2word = gensim.corpora.Dictionary(bigram)
        id2word.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        id2word.compactify()
        corpus = [id2word.doc2bow(text) for text in bigram]

        return corpus, id2word, bigram

    def fit(self, df):
        df_corpus, df_id2word, df_bigram = self.get_corpus(df, "review")

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lda = gensim.models.ldamulticore.LdaMulticore(
                corpus=df_corpus,
                num_topics=self.num_topics,
                id2word=df_id2word,
                per_word_topics=True)

        return lda, df_corpus, df_id2word, df_bigram

    def predict(self, df, lda, df_corpus):
        topic_vec = []
        for i in range(len(df)):
            top_topics = lda.get_document_topics(
                df_corpus[i], minimum_probability=self.min_prob)
            topic_values = sorted(top_topics, key=lambda x: x[1])[-1]
            topic_vec += [topic_values]

        topics = list(map(lambda x: str(x[0]), topic_vec))

        df["topic"] = topics

        return df

    def evaluate(self, df):
        topics = list(set(df["topic"]))
        topics.sort()
        fig = visualise_top_words(
            df, topics,
            custom_sw=self.common_words
        )
        return fig
