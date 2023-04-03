import pandas as pd
import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import nltk
from nltk.tokenize import word_tokenize
import joblib
import gensim
from gensim.models import Word2Vec
from src.models.classifier import Classifier
from src.utils.file_util import FileUtil

nltk.download('punkt')


class LOGREG(Classifier):
    def __init__(self, load_model=False):
        self.load_model = load_model
        self.saved_model_path = FileUtil().LOGREG_SENTIMENT_MODEL_DIR
        self.logreg_config = FileUtil.get_config()["LOGREG"]
        self.target_col = self.logreg_config["target_col"]
        self.text_col = self.logreg_config["text_col"]
        self.vector_size = self.logreg_config["vector_size"]
        self.window = self.logreg_config["window"]
        self.min_count = self.logreg_config["min_count"]
        self.sg = self.logreg_config["sg"]
        self.word_set = set()
        self.w2v_model = ""

        if load_model:
            if not FileUtil.check_dir_exists(self.saved_model_path):
                raise FileNotFoundError("There is no saved model in path",
                                        self.saved_model_path)
            self.model = joblib.load(self.saved_model_path)

    def tokenize(self, df):
        df['cleaned_text_new'] = df['cleaned_text'].apply(
            lambda x: word_tokenize(x))

        return df

    def train_w2v_model(self, train):

        X_train = train[self.text_col]

        w2v_model = gensim.models.Word2Vec(X_train,
                                           vector_size=self.vector_size,
                                           window=self.window,
                                           min_count=self.min_count,
                                           sg=self.sg)

        self.w2v_model = w2v_model

        return w2v_model

    def get_word_vectors(self, df):

        X_train = df[self.text_col]

        # words that appear in the train w2v model
        words = set(self.w2v_model.wv.index_to_key)
        self.word_set = words

        # train data
        X_vect = np.array([np.array(
            [self.w2v_model.wv[i] for i in ls if i in words])
            for ls in X_train], dtype=object)

        X_vect_avg = []
        for v in X_vect:
            if v.size:
                # take average weights across all the word
                # vectors within the sentence vector
                X_vect_avg.append(v.mean(axis=0))
            else:
                # else set zero vector of size 100 because the
                # size of vector that we initially set is 100
                X_vect_avg.append(np.zeros(self.vector_size,
                                           dtype=float))

        return X_vect_avg

    def fit(self, train):
        assert self.load_model is not True

        X_train_vect_avg = self.get_word_vectors(train)

        y_train = train[self.target_col]
        logr = LogisticRegression(random_state=1)
        logr.fit(X_train_vect_avg, y_train.values.ravel())
        self.model = logr

        return self.model

    def predict(self, valid):

        X_valid_vect_avg = self.get_word_vectors(valid)

        # Use the trained model to make predictions on the val data

        y_test = valid[self.target_col]
        LR_y_pred = self.model.predict_proba(X_valid_vect_avg)
        lr_y_test = y_test.copy()

        return LR_y_pred, lr_y_test

    def evaluate(self, valid):
        y_pred, y_label = self.predict(valid)

        # keep probabilities for the positive outcome only
        LR_y_probs = y_pred[:, 1]

        precision, recall, thresholds = precision_recall_curve(
            y_label, LR_y_probs
        )

        ap = average_precision_score(y_label,  LR_y_probs)
        pr_auc = auc(recall, precision)

        return ap, pr_auc
